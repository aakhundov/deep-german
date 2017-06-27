import textwrap


min_allowed_noun_length = 2
max_allowed_noun_length = 30
allowed_gender_labels = ["m", "f", "n"]
special_german_chars = list("äöüß")

raw_words_path = "./data/raw_words.txt"
short_nouns_path = "./data/short_nouns.txt"
raw_nouns_path = "./data/raw_nouns.txt"
clean_nouns_path = "./data/clean_nouns.txt"


print("Reading data...")
with open(raw_words_path, encoding="utf-8") as f:
    word_lines = f.readlines()

print("Reading short nouns...")
with open(short_nouns_path, encoding="utf-8") as f:
    short_nouns = [l[:-1] for l in f.readlines()]


def is_german_noun(word):
    # too short word
    if len(word) < min_allowed_noun_length:
        return False
    # too long word
    elif len(word) > max_allowed_noun_length:
        return False
    # short noun not in pre-filtered list
    elif len(word) < 4 and word not in short_nouns:
        return False
    # acronym
    elif str.isupper(word[1]):
        return False
    # first char lower-case
    elif not str.isupper(word[0]):
        return False
    else:
        for char in str.lower(word):
            code = ord(char)

            # code lower than [a-z]
            if code < 97:
                return False
            # code upper than [a-z] and not special German
            elif code > 122 and char not in special_german_chars:
                return False

    return True


print("Filtering raw nouns...")
# filtering out all non-noun words from the list of all words
all_noun_lines = [w for w in word_lines if w.endswith("noun\n")]

print("Writing raw nouns...")
# writing filtered lines into a file
with open(raw_nouns_path, "w+", encoding="utf-8") as f:
    f.writelines(all_noun_lines)


nouns_and_genders = {}
noun_characters = []

length_distribution = {l: 0 for l in range(min_allowed_noun_length, max_allowed_noun_length+1)}
gender_distribution = {g: 0 for g in allowed_gender_labels}

mixed_gender_words = 0
unrecognized_labels = 0
nouns_without_gender = 0
filtered_out_nouns = 0

print("Parsing nouns and genders...")
# parsing nouns and their genders
for i, nl in enumerate(all_noun_lines):
    if "{" in nl and "}" in nl:
        # positions of { and } in raw noun line
        pos1, pos2 = nl.index("{"), nl.index("}")

        # "noun" is the last word before opening {
        noun = nl[0:pos1 - 1].split(" ")[-1].split("-")[-1]
        # "gender" is the label between { and }
        gender = nl[pos1+1:pos2]

        # filtering out non-German nouns
        # according to criteria specified
        # in is_german_noun() function
        if not is_german_noun(noun):
            filtered_out_nouns += 1
            continue

        # only nouns with one of allowed
        # gender labels are considered
        if gender in allowed_gender_labels:
            # a noun observed for the first time
            if noun not in nouns_and_genders:
                nouns_and_genders[noun] = {gender: 1}
                gender_distribution[gender] += 1

                # updating the characters
                for c in noun:
                    if c not in noun_characters:
                        noun_characters.append(c)

                # updating length distribution
                length_distribution[len(noun)] += 1

            # same noun observed more than once
            else:
                genders_so_far = nouns_and_genders[noun].keys()
                # with the gender observed before
                if gender in genders_so_far:
                    nouns_and_genders[noun][gender] += 1
                # with the new gender (not observed before)
                else:
                    # second gender in a row
                    if len(genders_so_far) == 1:
                        mixed_gender_words += 1
                    nouns_and_genders[noun][gender] = 1
                    gender_distribution[gender] += 1
        else:
            unrecognized_labels += 1
    else:
        nouns_without_gender += 1


print("Writing nouns and genders...")
# writing cleaned-up list of nouns and
# their genders into a file
with open(clean_nouns_path, "w+", encoding="utf-8") as f:
    for noun in sorted(nouns_and_genders.keys()):
        gender_counts = ",".join([
            str(nouns_and_genders[noun][g])
            if g in nouns_and_genders[noun] else "0"
            for g in allowed_gender_labels
        ])
        f.write("{0}\t{1}\n".format(
            noun.lower(), gender_counts
        ))


print()
print("Statistics of the results:")
print("--------------------------")
print("Total words:", len(word_lines))
print("Total nouns:", len(all_noun_lines))
print("Unique nouns:", len(nouns_and_genders))
print("Mixed gender nouns:", mixed_gender_words)
print("Nouns without gender:", nouns_without_gender)
print("Filtered out nouns:", filtered_out_nouns)
print("Nouns with unrecognized gender labels:", unrecognized_labels)
print()

print("Gender distribution among recorded nouns:")
print("-----------------------------------------")
print(gender_distribution)

print()
print("Characters observed in all nouns:")
print("---------------------------------")
print("\n".join(textwrap.wrap(
    "".join(sorted(noun_characters)), 30
)))

print()
print("Length distribution:")
print("--------------------")
for l in sorted(length_distribution.keys()):
    print("{0}\t{1}".format(l, length_distribution[l]))
