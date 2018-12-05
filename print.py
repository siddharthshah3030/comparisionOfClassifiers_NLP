TP = cm[1][1]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[0][0]

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1_score = 2 * Precision * Recall / (Precision + Recall)

print('accuracy =', Accuracy, ' ,precision =',Precision,' ,Recall =', Recall, ' F1Score ' ,F1_score)

