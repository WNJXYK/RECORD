import torch
from torch import nn
import numpy as np
import torch.nn.functional as function
import torch.nn.functional as F
from sklearn import preprocessing


class LogisticRegression(nn.Module):
    def __init__(self, input_dims, output_dims, n_hidden = None):
        super(LogisticRegression, self).__init__()
        if n_hidden is None: n_hidden = int((input_dims + output_dims) * 3 / 2)
        self.hidden = torch.nn.Linear(input_dims, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, output_dims)
        self.linear = nn.Linear(input_dims, output_dims, bias=True)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

class MT():
    def __init__(self, classes, n_features, max_iter=1000, log=False, n_thread = 6):
        torch.set_num_threads(n_thread)
        torch.manual_seed(2147483647)

        self.teacher, self.student = None, None

        self.n_features = n_features
        classes = list(set(classes))
        self.classes, self.n_classes = np.array(classes), len(classes)
        self.rc = {}
        for i in range(self.n_classes): self.rc[classes[i]] = i

        self.max_iter = max_iter
        self.log = log

    def update_parameter(self, student, teacher, alpha):
        for teacher_parameter, param in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.data.mul_(alpha).add_(1 - alpha, param.data)

    def softmax_mse_loss(self, current, target):
        a = nn.functional.softmax(current, dim=1)
        b = nn.functional.softmax(target, dim=1)
        return function.mse_loss(a, b, reduction='mean') / current.size()[1]


    def fit(self, X, uX, _y):
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(np.vstack([X, uX]))
        X = self.scaler.transform(X)
        if uX.shape[0] > 0: uX = self.scaler.transform(uX)

        self.teacher = LogisticRegression(self.n_features, self.n_classes)
        self.student = LogisticRegression(self.n_features, self.n_classes)
        self.student.train()
        self.teacher, self.student = self.teacher.cuda(), self.student.cuda()
        for param in self.teacher.parameters(): param.detach_()

        classify_loss_function = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        classify_loss_function = classify_loss_function.cuda()
        optimizer = torch.optim.SGD(self.student.parameters(), lr=0.001, momentum=0.9)

        y = np.array([self.rc[_y[i]] for i in range(len(_y))])
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long()
        aX = torch.from_numpy(np.vstack([X, uX])).float()
        X, y, aX = X.cuda(), y.cuda(), aX.cuda()

        last_loss = None
        for epoch in range(self.max_iter):
            optimizer.zero_grad()

            pred_student = self.student(X)
            classify_loss = classify_loss_function(pred_student, y)

            pred_student = self.student(aX)
            pred_teacher = self.teacher(aX)
            consistency_loss = self.softmax_mse_loss(pred_student, pred_teacher)


            loss = classify_loss + consistency_loss
            if self.log and epoch % 1000 == 0: print(classify_loss, consistency_loss, loss)
            loss.backward()
            optimizer.step()

            alpha = min(1. - 1. / (epoch + 1), 0.99)
            self.update_parameter(self.student, self.teacher, alpha)


    def predict_proba(self, X):
        self.student.eval()
        X = self.scaler.transform(X)
        X = torch.from_numpy(X).float()
        X = X.cuda()
        pred_student = self.student(X)
        pred_proba = nn.functional.softmax(pred_student, dim=1)
        return pred_proba.detach_().cpu().numpy()


    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)].ravel()
