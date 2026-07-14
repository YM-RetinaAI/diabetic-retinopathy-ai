import torch

def simple_train_epoch(model, loader, optimizer, criterion):
    model.train()

    for images, true_labels in loader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, true_labels)
        loss.backward()
        optimizer.step()

@torch.no_grad() 
def simple_evaluate(model, loader, criterion):
    model.eval()
    all_answers = []
    all_true_labels = []
    for images, true_labels in loader:
        predictions = model(images)
        final_answer = predictions.argmax(dim=1)
        all_answers.extend(final_answer)
        all_true_labels.extend(true_labels)
    return calculate_score(all_true_labels, all_answers)