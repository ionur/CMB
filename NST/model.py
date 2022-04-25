import torch
import torch.nn as nn
import torch.nn.functional as F
########################################################################
# models
########################################################################



class LossNet(nn.Module):

    def __init__(self):
        super(LossNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)

        self.linear = nn.Linear(4608, 1)

        


    def forward(self, X, return_all=False):

        outputs = []
        
        X = self.conv1(X)
        X = self.bn1(X)
        X = torch.relu(X)
        outputs.append(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = F.max_pool2d(X, 3)
        outputs.append(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = torch.relu(X)
        outputs.append(X)

        X = self.conv4(X)
        X = self.bn4(X)
        X = F.max_pool2d(X, 2)
        outputs.append(X)

        X = self.conv5(X)
        X = self.bn5(X)
        X = torch.relu(X)
        outputs.append(X)

        X = self.conv6(X)
        X = self.bn6(X)
        X = F.max_pool2d(X, 2)
        outputs.append(X)

        X = self.conv7(X)
        X = self.bn7(X)
        X = torch.relu(X)
        outputs.append(X)

        X = self.conv8(X)
        X = self.bn8(X)
        X = torch.relu(X)
        outputs.append(X)

        X = torch.flatten(X, start_dim=1)
        X = self.linear(X)
        
        if return_all:
            return X, outputs
        
        else:
            return X



class Autoencoder(nn.Module):
    """
    the autoencoder class
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = Decoder()

        


    def forward(self, input, return_all=False):
        # encode input
        if return_all:
            input_encoded, Z = self.encoder(input, return_all=True)
        else:
            input_encoded = self.encoder(input, return_all=False)

        input_encoded = nn.functional.dropout(input_encoded, 0.2)
        # get output
        output = self.decoder(input_encoded)

        if return_all:
            return output, Z

        else:
            return output

   
class Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self):
        super(Decoder, self).__init__()

        filters = 8
        
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(128,  64, 3, 1, 0)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(64,  64, 3, 1, 0)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(64,  64, 3, 1, 0)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(64,  64, 3, 1, 0)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(64,  32, 3, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d( 32,  32, 3, 1, 0)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d( 32, 16, 3, 1, 0)
        self.bn4_1 = nn.BatchNorm2d(16)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(16, 1, 3, 1, 0)
        self.bn4_2 = nn.BatchNorm2d(1)

    def forward(self, input):
        # first block
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.bn1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.bn2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.bn2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.bn2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.bn2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.bn3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.bn3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.bn4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)
        out = self.bn4_2(out)

        return out


class Encoder(nn.Module):
    """
    the encoder network
    """
    def __init__(self):
        super(Encoder, self).__init__()

 
        # first block
        self.conv_1_1 = nn.Conv2d(1, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv_1_2 = nn.Conv2d(3, 32, 3, 1, 0)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(32, 64, 3, 1, 0)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(128, 256, 3, 1, 0)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu_4_1 = nn.ReLU(inplace=True)

    def forward(self, input, return_all=False):
        outputs = []

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.bn1_2(out)
        out = self.relu_1_2(out)
        outputs.append(out)

        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.bn1_3(out)
        out = self.relu_1_3(out)

        out = self.maxPool_1(out)
        outputs.append(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.bn2_1(out)
        out = self.relu_2_1(out)
        outputs.append(out)

        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.bn2_2(out)
        out = self.relu_2_2(out)

        out = self.maxPool_2(out)
        outputs.append(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.bn3_1(out)
        out = self.relu_3_1(out)
        outputs.append(out)

        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.bn3_2(out)
        out = self.relu_3_2(out)
        outputs.append(out)

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.bn3_3(out)
        out = self.relu_3_3(out)
        outputs.append(out)

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.bn3_4(out)
        out = self.relu_3_4(out)

        out = self.maxPool_3(out)
        outputs.append(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.bn4_1(out)
        out = self.relu_4_1(out)
        outputs.append(out)

        if return_all:
            return out, outputs
        else:
            return out


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(32, 64, 128, 256, 512)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet2D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0], dropout=0.2))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1], dropout=0.2)
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i], dropout=0.2)
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)
       #  self.conv_out = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 2), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 3, 1, 1))



    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))


        # out = self.conv_out(x_dec[-1])

        # x_dec.append(self.smooth_conv(x_dec[-1]))
        if not return_all: 
            return x_dec[-1]
        else:
            return x_enc + x_dec

class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            # nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)

def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)