import numpy as np


class Trainer:
    
    def __init__(self, model, loss_fn, evaluator_fn, model_activation, optimizer, optimizer_args):
        self.model = model
        self.loss_fn = loss_fn
        self.evaluator_fn = evaluator_fn
        self.model_activation = model_activation
        self.optimizer = optimizer(model.params(), **optimizer_args)

        self.train_losses = []
        self.train_scores = []
        self.val_losses = []
        self.val_scores = []

    def train(self, train_dataloader, stop_condition, stop_condition_args, val_dataloader=None):
        while not stop_condition(self.train_losses, self.train_scores, self.val_losses, self.val_scores, **stop_condition_args):
            self.train_epoch(train_dataloader)
            train_loss, train_predictions, train_labels = self.compute_loss(train_dataloader, return_predictions=True)
            self.train_losses.append(train_loss)
            self.train_scores.append(self.evaluate_fn(train_predictions, train_labels))
            if val_dataloader is not None:
                val_loss, val_predictions, val_labels = self.compute_loss(val_dataloader, return_predictions=True)
                self.val_losses.append(val_loss)
                self.val_scores.append(self.evaluator_fn(val_predictions, val_labels))


    def train_epoch(self, dataloader):
        for inpts, labels in dataloader:
            logits = self.model(inpts)
            loss = self.loss_fn(logits, labels)
            loss.zero_grads()
            loss.backward()
            self.optimizer.update()

    def compute_loss(self, dataloader, return_predictions):
        total_loss = 0
        total_samples = 0
        prediction_list = []
        label_list = []
        for inpts, labels in dataloader:
            batch_size = inpts.shape[0]
            logits = self.model(inpts)
            loss = self.loss_fn(logits, labels)
            total_loss += loss
            total_samples += batch_size
            if return_predictions:
                prediction_list.append(self.model_activation(logits).elems)
                label_list.append(labels.elems)
        if return_predictions:
            return total_loss/total_samples, np.concatenate(prediction_list), np.concatenate(label_list)
        else:
            return total_loss/total_samples

    def get_predictions(self, dataloader):
        prediction_list = []
        label_list = []
        for inpts, labels in dataloader:
            logits = self.model(inpts)
            batch_predictions = self.model_activation(logits).elems
            prediction_list.append(batch_predictions)
            label_list.append(labels.elems)
        return np.concatenate(prediction_list), np.concatenate(label_list)

    def evaluate(self, dataloader):
        predictions, labels = self.get_predictions(dataloader)
        return self.evaluator_fn(predictions, labels)


def max_epochs(train_losses, train_scores, val_losses, val_scores, num_epochs):
    return len(train_losses) >= num_epochs


def val_score_reduction(train_losses, train_scores, val_losses, val_scores, mean_range):
    if len(val_scores) <= mean_range:
        return False
    current_mean = sum(val_scores[-mean_range:])/mean_range
    prev_mean = sum(val_scores[(-mean_range-1):-1])/mean_range
    return current_mean < prev_mean


def prf(predictions, labels):
    delta = 10**(-5)
    label_set = np.arange(predictions.shape[1])
    predicted_labels = np.argmax(predictions, axis=1)
    scores = {}
    for label in label_set:
        TP = sum(np.logical_and(predicted_labels == label, labels == label))
        FP = sum(np.logical_and(predicted_labels == label, np.logical_not(labels == label)))
        FN = sum(np.logical_and(np.logical_not(predicted_labels == label), labels == label))
        precision = TP/(TP+FP+delta)
        recall = TP/(TP+FN+delta)
        f_score = (2*precision*recall)/(precision+recall+delta)
        scores[label] = {'precision': precision, 'recall': recall, 'f-score': f_score}
    return scores


def accuracy(predictions, labels):
    return sum(np.argmax(predictions.elems, axis=1) == labels.elems)/len(labels)
