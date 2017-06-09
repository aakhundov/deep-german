import textwrap


special_german_chars = list("äöüß")

print("Reading data...")
with open("raw_words.txt", encoding="utf-8") as f:
    word_lines = f.readlines()

print("Reading short nouns...")
with open("short_nouns.txt", encoding="utf-8") as f:
    short_nouns = [l[:-1] for l in f.readlines()]


def is_german_noun(word):
    # empty word
    if len(word) < 2:
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
with open("raw_nouns.txt", "w+", encoding="utf-8") as f:
    f.writelines(all_noun_lines)


nouns_and_genders = {}
gender_distribution = {}
characters = []

allowed_gender_labels = ["m", "f", "n", "pl"]

mixed_gender_words = 0
unrecognized_labels = 0
nouns_without_gender = 0
filtered_out_nouns = 0
min_noun_length = 1000
max_noun_length = 0
shortest_noun = ""
longest_noun = ""

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
            # initializing gender counter
            if gender not in gender_distribution:
                gender_distribution[gender] = 0

            # a noun observed for the first time
            if noun not in nouns_and_genders:
                nouns_and_genders[noun] = [gender]
                gender_distribution[gender] += 1

                # registering the characters
                for c in noun:
                    if c not in characters:
                        characters.append(c)

                # comparing the length
                noun_length = len(noun)
                if noun_length > max_noun_length:
                    max_noun_length = noun_length
                    longest_noun = noun
                if noun_length < min_noun_length:
                    min_noun_length = noun_length
                    shortest_noun = noun

            # same noun observed with different gender
            elif gender not in nouns_and_genders[noun]:
                if len(nouns_and_genders[noun]) == 1:
                    mixed_gender_words += 1
                nouns_and_genders[noun].append(gender)
                gender_distribution[gender] += 1
        else:
            unrecognized_labels += 1
    else:
        nouns_without_gender += 1


print("Writing nouns and genders...")
# writing cleaned-up list of nouns and
# their genders into a file
with open("clean_nouns.txt", "w+", encoding="utf-8") as f:
    for noun in sorted(nouns_and_genders.keys()):
        gender = ",".join(sorted(nouns_and_genders[noun]))
        f.write("{0}\t{1}\n".format(
            noun.lower(), gender
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
print("Nouns length range:", min_noun_length, "-", max_noun_length)
print("Nouns with unrecognized gender labels:", unrecognized_labels)
print()

print("Shortest noun:")
print(shortest_noun)
print("Longest noun:")
print(longest_noun)
print()

print("Gender distribution among recorded nouns:")
print(gender_distribution)

print()
print("Characters observed in all nouns:")
print("---------------------------------")
print("\n".join(textwrap.wrap(
    "".join(sorted(characters)), 30
)))
