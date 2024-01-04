glove_embeddings = {}

with open('glove_50dim_wiki.id.case.text.txt', encoding='utf8') as f:
    for line in f:
        try:
            line = line.split()
            glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
        except:
            continue
            
