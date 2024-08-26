from .train_types import NodeAttribute

# simplified weights
# TODO: experimental support for deciding the weight
similarity_reference = {
    NodeAttribute.PROCESSOR: 5,
    NodeAttribute.CORES: 1,
    NodeAttribute.CHIPS: 1,
    NodeAttribute.MEMORY: 0.5,
    NodeAttribute.FREQ: 0.5,
}

similarity_total_weight = sum(similarity_reference.values())

def get_similarity_weight(attr):
    return similarity_reference[attr]/similarity_total_weight

def compute_jaccard_similarity(str1 :str, str2: str) -> float:
    if str1.lower() == str2.lower(): # including the case of both are empty
        return 1
    if len(str1) == 0 or len(str2)==0:
       return 0
    set1 = set(str1.lower())  # Convert to lowercase for case-insensitive comparison
    set2 = set(str2.lower())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    similarity = intersection / union * 0.5
    return similarity

def compute_similarity(base : float, cmp: float) -> float:
    base = float(base)
    cmp = float(cmp)
    diff_ratio = 0
    if base > 0 or cmp > 0:
        diff_ratio = abs(cmp-base)/((base+cmp)/2)
    if diff_ratio >= 1:
        return 0
    else:
        return 1-diff_ratio

def compute_looseness(similarity):
    return 1-similarity

# get_candidate_score returns certainty
def get_candidate_score(candidate_uncertain_attribute_freq, candidate_uncertain_attribute_total):
    candidate_score = dict()
    for attr, candidates in candidate_uncertain_attribute_freq.items():
        total = candidate_uncertain_attribute_total[attr]
        if total == 0:
            # no uncertainty
            continue
        for candidate in candidates:
            candidate_index = candidate[0]
            candidate_freq = candidate[1]
            if candidate_index not in candidate_score:
                candidate_score[candidate_index] = 0
            candidate_score[candidate_index] += float(candidate_freq)/total
    return candidate_score

def find_best_candidate(candidate_score):
    max_score = 0
    best_candidate_index = -1
    for index, score in candidate_score.items():
        if score > max_score:
            best_candidate_index = index
            max_score = score
    return best_candidate_index, max_score

def compute_uncertainty(max_score, num_of_none):
    if num_of_none == 0:
        return 0 # covered
    uncertainty = 1 - max_score/num_of_none
    return uncertainty

def get_num_of_none(in_spec):
    num_of_none = 0
    for attr in NodeAttribute:
        if in_spec.attrs[attr] is None:
            num_of_none += 1
    return num_of_none
