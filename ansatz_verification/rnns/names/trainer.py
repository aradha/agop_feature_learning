import random
import torch
import torch.nn as nn


def train(rnn, category_tensor, input_line_tensor, target_line_tensor, learning_rate, num_categories):
    criterion = nn.NLLLoss()
    #criterion = nn.MSELoss()

    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden().to(category_tensor.device)

    rnn.zero_grad()

    loss = torch.Tensor([0]).to(category_tensor.device) # you can also just simply use ``loss = 0``

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        #y = torch.zeros_like(output)
        #y[:,target_line_tensor[i]] = 1
        #l = criterion(output, y)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def gru_train(rnn, category_tensor, input_line_tensor, target_line_tensor, learning_rate, num_categories):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass to get output/logits
    # outputs.size() --> 100, 10
    outputs = rnn(category_tensor, input_line_tensor, target_line_tensor)

    # Calculate Loss: softmax --> cross entropy loss
    loss = criterion(outputs, target_line_tensor)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    return hidden, outputs
