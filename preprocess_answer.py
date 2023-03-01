# %%
def bio(chars, tag):
    tag = tag.upper()
    if tag == 'O':
        tags = [tag] * len(chars)
    else:
        tags = ['B-'+tag] + ['I-'+tag] * (len(chars) - 1)
    return list(zip(chars, tags))

def bioes(chars, tag):
    tag = tag.upper()
    if tag == 'O':
        tags = [tag] * len(chars)
    elif len(chars) == 1:
        tags = ['S-'+tag]
    else:
        tags = ['B-'+tag] + ['I-'+tag] * (len(chars) - 2) + ['E-'+tag]
    return list(zip(chars, tags))
# %%
def process(file, target, mode = "bio"):
    if mode == "bio":
        labeling = bio
    elif mode == "bioes":
        labeling = bioes
    else:
        raise NotImplementedError
    with open(file, encoding="utf-8")as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_line = []
        line = line.strip()
        words = line.split(" ")
        for word in words:
            chars, tag = word.split("/")
            new_line += labeling(chars, tag)
        new_lines.append(new_line)
    with open(target, "w", encoding="utf-8")as fw:
        fw.writelines("\n\n".join(["\n".join([" ".join(pair) for pair in line]) for line in new_lines]))


# %%
if __name__ == "__main__":
    process("./data/raw/train1.txt", "./data/processed/train1_bio.txt", "bio")
    process("./data/raw/train1.txt", "./data/processed/train1_bioes.txt", "bioes")
    process("./data/raw/testright1.txt", "./data/processed/testright_bio.txt", "bio")
    process("./data/raw/testright1.txt", "./data/processed/testright_bioes.txt", "bioes")

# %%
