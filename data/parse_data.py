import textwrap


print("Reading data...")
with open("raw_words.txt", encoding="utf-8") as f:
    word_lines = f.readlines()

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
min_noun_length = 0
max_noun_length = 0

print("Parsing nouns and genders...")
# parsing nouns and their genders
for i, nl in enumerate(all_noun_lines):
    if "{" in nl and "}" in nl:
        # positions of { and } in raw noun line
        pos1, pos2 = nl.index("{"), nl.index("}")

        # "noun" is the last word before opening {
        noun = nl[0:pos1 - 1].split(" ")[-1]
        # "gender" is the label between { and }
        gender = nl[pos1+1:pos2]

        # TODO: any pre-processing and further filtering out
        # TODO: can be done here with "continue" operator

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
                if noun_length < min_noun_length:
                    min_noun_length = noun_length

            # same noun observed with different gender
            elif gender not in nouns_and_genders[noun]:
                nouns_and_genders[noun].append(gender)
                mixed_gender_words += 1
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
        f.write("{0}\t{1}\n".format(
            noun, ",".join(sorted(nouns_and_genders[noun]))
        ))


print()
print("Statistics of the results:")
print("---------------------------------")
print("Total words:", len(word_lines))
print("Total nouns:", len(all_noun_lines))
print("Unique nouns:", len(nouns_and_genders))
print("Mixed gender nouns:", mixed_gender_words)
print("Nouns without gender:", nouns_without_gender)
print("Nouns length range:", min_noun_length, "-", max_noun_length)
print("Nouns with unrecognized gender labels:", unrecognized_labels)
print("Gender distribution among recorded nouns:")
print(gender_distribution)

print()
print("Characters observed in all nouns:")
print("---------------------------------")
print("\n".join(textwrap.wrap(
    "".join(sorted(characters)), 30
)))
