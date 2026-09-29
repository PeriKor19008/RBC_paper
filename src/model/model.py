import torch
import torch.nn as nn


class multyLabelCNN(nn.Module):
    def __init__(self,  conv_config=[("conv", 16), ("conv", 32)], fc_config=[128]):
        super(multyLabelCNN, self).__init__()
        input_shape = (1, 50, 50)

        output_dim = 1
        self.conv_layers = nn.Sequential()
        in_channels = input_shape[0]
        h, w = input_shape[1], input_shape[2]

        # === Build convolutional layers ===

        for idx, (layer_type, out_channels) in enumerate(conv_config):
            if layer_type != "conv":
                raise ValueError(f"Unsupported conv layer type: {layer_type}")


            self.conv_layers.add_module(
                f"conv{idx}",
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.add_module(f"relu{idx}", nn.ReLU(inplace=True))

            # 2) add pooling only for blocks 1..4 (i.e., after the first conv, up to the 5th conv)
            if 1 <= idx <= 4 and min(h, w) >= 2:
                self.conv_layers.add_module(f"pool{idx}", nn.MaxPool2d(2))
                h //= 2
                w //= 2

            in_channels = out_channels

        # === Flatten layer before FC ===

        flat_size = in_channels * h * w

        # === Build fully connected layers ===
        fc_layers = []
        fc_in = flat_size

        for idx, fc_out in enumerate(fc_config):
            fc_layers.append(nn.Linear(fc_in, fc_out))
            fc_layers.append(nn.ReLU())
            fc_in = fc_out

        fc_layers.append(nn.Linear(fc_in, output_dim))  # Final output layer

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x








