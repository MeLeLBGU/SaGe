
from scipy.special import expit
import numpy as np
import math

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
WORD_PREFIX_CHAR = "\u2581"

def sigmoid(num):
    s = expit(num)
    if s == 1.0:
        raise BaseException("sigmoid overflow")

    return expit(num)

def compute_window(token_index, tokens_in_line, window_size):
    # get context window tokens
    context_start = max(token_index-window_size, 0)
    context_end = min(token_index+window_size+1, len(tokens_in_line))
    window = tokens_in_line[context_start:token_index] + tokens_in_line[token_index+1:context_end]

    return window, context_start, context_end

# Compose list of tokens we never ablate
def get_not_ablateable_tokens_list(current_vocab):
    NAT_list = []
    for t in current_vocab:
        if len(t) == 1:
            NAT_list.append(t)
            continue

        if t.startswith(WORD_PREFIX_CHAR) and len(t) == 2:
            NAT_list.append(t)
            continue

        if t in SPECIAL_TOKENS:
            NAT_list.append(t)
            continue

        continue
        
    return NAT_list

# calculating 'offset' = number of tokens added because of our new encoding
def calculate_token_offset(i, last_times_accumulate_offset, tokens_in_line, updated_tokens_in_line, log):
    # original encoding next_word_index
    original_current_index = i+1
    if original_current_index == len(tokens_in_line):
        original_current_index = i
    else:
        while not tokens_in_line[original_current_index].startswith("\u2581"):
            original_current_index += 1
            if original_current_index == len(tokens_in_line):
                break
    
    original_next_word_index = original_current_index

    # and the updated next_word_index
    current_index = last_times_accumulate_offset + i+1
    if current_index == len(updated_tokens_in_line):
        current_index = last_times_accumulate_offset + i
    else:
        if (current_index >= len(updated_tokens_in_line)):
            log.info("Fail! current index = {}, length of \"updated_tokens_in_line\" = {}".format(current_index, len(updated_tokens_in_line)))
            log.info("token #{}, updated tokens:{}".format(i, updated_tokens_in_line))
            current_index -= 1
        else:
            while not updated_tokens_in_line[current_index].startswith("\u2581"):
                current_index += 1
                if current_index == len(updated_tokens_in_line):
                    break

    updated_next_word_index = current_index

    # the offset is the difference between them
    offset = updated_next_word_index - original_next_word_index - last_times_accumulate_offset

    return offset

def token_to_line_indices_dictionary(current_vocab, corpus_lines):
    # first lets hold the instance of line with "/u2581" chars instead of spaces
    corpus_lines_encoded = []
    for line in corpus_lines:
        enc_line = WORD_PREFIX_CHAR + line
        enc_line = enc_line.replace(" ", WORD_PREFIX_CHAR)
        corpus_lines_encoded.append(enc_line)

    # and test whether tokens contained in the lines
    token_to_line_dict = {}
    for token in current_vocab:
        lines_with_token_indices = []
        for i, enc_line in enumerate(corpus_lines_encoded):
            if token not in enc_line:
                continue
            lines_with_token_indices.append(i)
        token_to_line_dict[token] = lines_with_token_indices

    return token_to_line_dict

####################################################################
## Functions for multiprocessing - cannot be class methods #########
####################################################################

def sg_for_window_mp(target_token, window, target_embeddings, context_embeddings, log):
    current_p = 0
    for w in window:
        # calculate current value to add
        dot_product = np.dot(target_embeddings[target_token], context_embeddings[w])
        try:
            current_p += math.log(sigmoid(dot_product))
        except:
            pass

    return (-1) * current_p


def substract_windows_from_sg_mp(i, tokens_in_line_ints, current_sg, target_embeddings, context_embeddings, window_size, log):
    # we should substract windows of all tokens in our original window
    sg_wo = current_sg
    _, context_start, context_end = compute_window(i, tokens_in_line_ints, window_size)

    for index in range(context_start, context_end):
        window, _, _ = compute_window(index, tokens_in_line_ints, window_size)
        sg_wo -= sg_for_window_mp(tokens_in_line_ints[index], window, target_embeddings, context_embeddings, log)

    return sg_wo

def add_windows_to_sg_mp(model, updated_i, offset, updated_context_start, \
                        updated_context_end, updated_tokens_in_line, current_sg, \
                        target_embeddings, context_embeddings, window_size, log):
    sg_wo = current_sg
    
    # assume it now looks like: old[context_start:i] + [ablated_token_encoding] + old[i+1:context_end]
    # add the windows before i (ablated token)
    for index in range(updated_context_start, updated_i):
        try:
            window, _, _ = compute_window(index, updated_tokens_in_line, window_size)
            sg_wo += sg_for_window_mp(updated_tokens_in_line[index], window, target_embeddings, context_embeddings, log)
        except:
            print("index  {}".format(index))
            print("tokens {}".format(updated_tokens_in_line))
            raise

    # add the ablated token encoding windows
    for index in range(updated_i, updated_i+offset):
        try:
            window, _, _ = compute_window(index, updated_tokens_in_line, window_size)
            sg_wo += sg_for_window_mp(updated_tokens_in_line[index], window, target_embeddings, context_embeddings, log)
        except:
            print("index  {}".format(index))
            print("tokens {}".format(updated_tokens_in_line))
            raise

    # add the windows after the ablated token
    for index in range(updated_i+offset, updated_context_end):
        try:
            window, _, _ = compute_window(index, updated_tokens_in_line, window_size)
            sg_wo += sg_for_window_mp(updated_tokens_in_line[index], window, target_embeddings, context_embeddings, log)
        except:
            print("index  {}, len {}".format(index, len(updated_tokens_in_line)))
            updated_tokens_in_line_str = [model.id_to_piece(x) for x in updated_tokens_in_line]
            print("tokens {}".format(updated_tokens_in_line_str))
            raise

    return sg_wo

def update_sg_per_instance_of_token_mp(model, token_to_ablate, i, \
                                    line, tokens_in_line_ints, tokens_in_line_pieces, \
                                    last_times_accumulate_offset, \
                                    current_total_sg, current_vocab, \
                                    target_embeddings, context_embeddings, log, window_size):
    
    # window of token - before ablation
    sg_wo = current_total_sg

    # we should substract windows of all tokens in our original window
    original_window, original_context_start, original_context_end = compute_window(i, tokens_in_line_pieces, window_size)

    sg_wo = substract_windows_from_sg_mp(i, tokens_in_line_ints, sg_wo, target_embeddings, context_embeddings, window_size, log)

    ## remove token from current vocab
    vocab_without_token = current_vocab.copy()
    vocab_without_token.remove(token_to_ablate)
    model.set_vocabulary(vocab_without_token)

    # encode with new vocab
    updated_tokens_in_line_ints = model.encode(line, out_type=int)
    updated_tokens_in_line = [model.id_to_piece(u) for u in updated_tokens_in_line_ints]

    # calculating 'offset' = number of tokens added because of our new encoding
    offset = calculate_token_offset(i, last_times_accumulate_offset, tokens_in_line_pieces, updated_tokens_in_line, log)

    #################################################################################
    # make sure the window (before the ablated token) stays the same ################
    #################################################################################
    _, updated_context_start, updated_context_end = compute_window(i+last_times_accumulate_offset, updated_tokens_in_line_ints, window_size)
    updated_context_end = min(updated_context_end+offset, len(updated_tokens_in_line)) # real context_en
    updated_i = i + last_times_accumulate_offset
    updated_window = updated_tokens_in_line[updated_context_start:updated_i] + updated_tokens_in_line[updated_i+offset+1:updated_context_end]

    first_part_of_window_length = min(window_size, updated_i)

    if updated_window[:first_part_of_window_length] != original_window[:first_part_of_window_length]:
        return current_total_sg, offset

    #################################################################################
    #################################################################################

    # and add the updated windows sg
    sg_wo = add_windows_to_sg_mp(model, updated_i, offset, updated_context_start, \
                                updated_context_end, updated_tokens_in_line_ints, \
                                sg_wo, target_embeddings, context_embeddings, window_size, log)

    ## revert model to current vocab
    model.set_vocabulary(current_vocab)

    return sg_wo, offset

## We assume the line contains the token_to_ablate (check it before shooting new process to execute this method)
def get_diff_sg_wo_token_for_line(model, line_index, \
                                    corpus_lines, model_encoded_corpus_lines_token_ids, model_encoded_corpus_lines_token_pieces, \
                                    token_to_ablate, current_vocab, target_embeddings, context_embeddings, log, window_size):
    sg_wo_diff = 0

    tokens_in_line_ints = model_encoded_corpus_lines_token_ids[line_index]
    tokens_in_line_pieces = model_encoded_corpus_lines_token_pieces[line_index]
    line = corpus_lines[line_index]
    
    # For start, assume there are no overlaps in the windows of 2 instances
    indices = [i for i, x in enumerate(tokens_in_line_pieces) if x == token_to_ablate]

    last_time_offset = 0
    last_times_accumulate_offset = 0
    for i in indices:
        sg_wo_diff, last_time_offset = update_sg_per_instance_of_token_mp(model, token_to_ablate, i, line, tokens_in_line_ints, tokens_in_line_pieces, last_times_accumulate_offset, sg_wo_diff, current_vocab, target_embeddings, context_embeddings, log, window_size)
        last_times_accumulate_offset += last_time_offset

    return token_to_ablate, sg_wo_diff

def sg_wo_token_mp(model, token_to_ablate, current_total_sg, current_vocab, training_filepath, target_embeddings, context_embeddings, log, corpus_lines, window_size):
    sg_wo = current_total_sg

    for line in corpus_lines:
        tokens_in_line_pieces = [model.id_to_piece(x) for x in model.encode(line, out_type=int)]
        if token_to_ablate not in tokens_in_line_pieces:
            continue

        # For start, assume there are no overlaps in the windows of 2 instances
        indices = [i for i, x in enumerate(tokens_in_line_pieces) if x == token_to_ablate]

        last_time_offset = 0
        last_times_accumulate_offset = 0
        for i in indices:
            tokens_in_line_ints = [model.piece_to_id(t) for t in tokens_in_line_pieces]
            sg_wo, last_time_offset = update_sg_per_instance_of_token_mp(model, token_to_ablate, i, line, tokens_in_line_ints, tokens_in_line_pieces, last_times_accumulate_offset, sg_wo, current_vocab, target_embeddings, context_embeddings, log, window_size)
            last_times_accumulate_offset += last_time_offset

    return token_to_ablate, sg_wo
