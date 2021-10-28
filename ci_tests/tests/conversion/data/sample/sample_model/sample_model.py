import torch
import numpy as np


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 10, 100, 10


class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


DYNAMIC_INPUT_SHAPE = [-1, D_in]
DYNAMIC_OUTPUT_SHAPE = [-1, D_out]
SAMPLE_INPUT_DATA_SHAPE = [N, D_in]
INPUT_NAME = "input_0"
OUTPUT_NAME = "output_0"


def _train_simple_model():
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = 2 * x + 10
    # y = torch.randn(N, D_out)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet()

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def save_entire_model(model_path):
    model = _train_simple_model()
    torch.save(model, model_path)


def save_sample_input_data(npz_file_path):
    data = {INPUT_NAME: torch.randn(SAMPLE_INPUT_DATA_SHAPE).numpy()}
    np.savez(npz_file_path, **data)


def get_input_shape(dynamic=True):
    return DYNAMIC_INPUT_SHAPE if dynamic else SAMPLE_INPUT_DATA_SHAPE


def get_output_shape(dynamic=True):
    return DYNAMIC_OUTPUT_SHAPE if dynamic else SAMPLE_INPUT_DATA_SHAPE
