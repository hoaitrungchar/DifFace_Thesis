import torch
import torch.nn as nn
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUnet, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        self.up1_convtrans = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_doubleconv = DoubleConv(1024, 512)

        self.up2_convtrans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_doubleconv = DoubleConv(512, 256)

        self.up3_convtrans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_doubleconv = DoubleConv(256, 128)

        self.up4_convtrans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_doubleconv = DoubleConv(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)        
        x2 = self.down1(x1)      
        x3 = self.down2(x2)     
        x4 = self.down3(x3)      
        x5 = self.down4(x4)     

        # Decoding path
        x = self.up1_convtrans(x5)            
        x = torch.cat([x4, x], dim=1)                
        x = self.up1_doubleconv(x)         

        x = self.up2_convtrans(x)                   
        x = torch.cat([x3, x], dim=1)           
        x = self.up2_doubleconv(x)                 

        x = self.up3_convtrans(x)                  
        x = torch.cat([x2, x], dim=1)             
        x = self.up3_doubleconv(x)                   

        x = self.up4_convtrans(x)                   
        x = torch.cat([x1, x], dim=1)               
        x = self.up4_doubleconv(x)                    
        # Output layer
        logits = self.outc(x)             
        return logits
