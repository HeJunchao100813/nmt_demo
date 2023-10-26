import sacrebleu


pred_bpe_path ="/home/hejunchao/Downloads/230_test/pred.txt"
target_bpe_path ="/home/hejunchao/Downloads/230_test/target.txt"

with open(pred_bpe_path, "w") as f:
    f.writelines(pred1)
with open(target_bpe_path, "w") as f:
    f.writelines(pred2)

# ###step3：计算评估指标
pred = []
tgt = []
candidates =[]
references=[]
with open(pred_bpe_path, "r") as f:
    for line in f.readlines():
        s = line.strip('\n')
        # print(s)
        pred.append(s)
        s = line.split()
        candidates.append([s])

with open(target_bpe_path, "r") as f:
    for line in f.readlines():
        s = line.strip('\n')
        tgt.append(s)
        s = line.split()
        references.append(s)

bleu = sacrebleu.corpus_bleu(pred[:100], [tgt[:100]], tokenize='zh')
print("bleu.score（zh）:", float(bleu.score))
print( bleu)
output=str(bleu.score)


pred_bpe_path="/data/hesha/fairseq_new2/reference_result/wmt19_test_big_100/EN_corpus_trans_1bs_test_beam3_kvcache.txt"
target_bpe_path="/data/hesha/fairseq_new2/reference_result/wmt19_test_big_100/target.txt"


pred = []
tgt = []
candidates =[]
references=[]
with open(pred_bpe_path, "r") as f:
    for line in f.readlines():
        s = line.strip('\n')
        # print(s)
        pred.append(s)
        s = line.split()
        candidates.append([s])

with open(target_bpe_path, "r") as f:
    for line in f.readlines():
        s = line.strip('\n')
        tgt.append(s)
        s = line.split()
        references.append(s)

bleu = sacrebleu.corpus_bleu(pred, [tgt[:100]], tokenize='zh')
print("bleu.score（zh）:", float(bleu.score))
print( bleu)
output=str(bleu.score)


