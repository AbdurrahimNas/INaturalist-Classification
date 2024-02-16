
import torch
from tqdm.auto import tqdm

results = {"train_loss": [],
          "train_acc": [],
          "test_loss": [],
          "test_acc": []
          }
def train_model(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          EPOCHS,
          device):
  for epoch in tqdm(range(0, EPOCHS)):


    model.train()
    train_loss, train_acc = 0,0
    for batch, (X,y) in enumerate(train_dataloader):
      X, y = X.to(device), y.to(device)
      y_preds = model(X)
      loss = loss_fn(y_preds, y)
      train_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item() / len(y_preds)

    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
      for batch, (X,y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        test_preds = model(X)
        tst_loss = loss_fn(test_preds, y)
        test_loss += tst_loss.item()

        test_pred_class = test_preds.argmax(dim=1)
        test_acc += (test_pred_class==y).sum().item() / len(test_preds)

      test_loss = test_loss / len(test_dataloader)
      test_acc = test_acc / len(test_dataloader)

    print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
