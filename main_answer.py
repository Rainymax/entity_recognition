# %%
from data_utils_answer import sent2features, sent2labels, sent2tokens, read_examples_from_file
import pycrfsuite
from nervaluate import Evaluator

mode = "bio"

train_file = "./data/processed/train1_bio.txt"
test_file = "./data/processed/testright_bio.txt"
model = "./msra_bio.crfsuite"
# %%
if __name__ == "__main__":
    # read data
    trainset = read_examples_from_file(train_file, "train")
    testset = read_examples_from_file(test_file, "test")


    X_train = [sent2features(s) for s in trainset]
    y_train = [sent2labels(s) for s in trainset]

    X_test = [sent2features(s) for s in testset]
    y_test = [sent2labels(s) for s in testset]
    # %%
    # training
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 500,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    print("training")
    import time
    tic = time.time()
    trainer.train(model)
    toc = time.time()
    print("done, total time:{toc-tic}")
    # %%
    # evaluation
    print("evaluation")
    tagger = pycrfsuite.Tagger()
    tagger.open(model)

    labels = [sent2labels(s) for s in testset]
    pred = [tagger.tag(sent2features(s)) for s in testset]

    tags = []
    for label in labels:
        tags += [l.split("-")[-1] for l in label]
    tags = list(set(tags))
    tags.remove("O")
    evaluator = Evaluator(labels, pred, tags=tags, loader="list")

    results, results_by_tag = evaluator.evaluate()
    # %%
    print(results_by_tag)
    print(results)
