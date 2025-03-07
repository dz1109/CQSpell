import sys

def strB2Q(ustring):
    """把字符串全角转半角"""
    result = []
    for char in ustring:
        code = ord(char)
        if code == 0x3000:  # 全角空格直接转换
            code = 0x0020
        elif 0xFF01 <= code <= 0xFF5E:  # 全角字符（除空格）转换公式
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)
def eval_model_batch(source_file, predict_file):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    query = []
    target = []
    predict = []
    with open(source_file) as fp:
        for line in fp:
            line = line.strip("\n").split("\t")
            query.append(line[0])
            target.append(strB2Q(line[1]))

    with open(predict_file) as fp:
        for line in fp:
            line = line.strip("\n").split("\t")
            predict.append(strB2Q(line[1]))

    for tgt_pred, src, tgt in zip(predict, query, target):
        if src == tgt:
            if tgt == tgt_pred:
                TN += 1
            else:
                FP += 1
        else:
            if tgt == tgt_pred:
                TP += 1
            else:
                FN += 1
        total_num += 1
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, total num: {total_num}')
    return acc, precision, recall, f1

if __name__ == "__main__":
    eval_model_batch(sys.argv[1], sys.argv[2])