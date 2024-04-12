import torch


class DCTTransformer:
    def __init__(self, height):
        """
        初始化类时，预先计算DCT矩阵
        """
        self.height = height
        self.dct_matrix = self._dct_matrix(height)


    def _dct_matrix(self, N):
        """
        Calculate the DCT matrix
        """
        n = torch.arange(N).float()
        k = torch.arange(N).float()
        cosines = torch.cos((2 * n + 1) * k * torch.pi / (2 * N))
        cosines[0] /= torch.sqrt(torch.tensor(2.0))
        cosines = torch.unsqueeze(cosines, dim=1)
        return cosines

    def dct(self, img):
        """
        DCT operation
        """
        batch_size, num_channels, height, width = img.size()
        DCT_matrix = self.dct_matrix

        img = img.permute(0, 2, 3, 1)
        img_dct = torch.zeros_like(img)

        for i in range(batch_size):
            for j in range(num_channels):
                img_channel = img[i, :, :, j].reshape(height, -1)
                img_dct[i, :, :, j] = torch.matmul(DCT_matrix.T, torch.matmul(img_channel, DCT_matrix))

        img_dct = img_dct.permute(0, 3, 1, 2)
        return img_dct.float()

    def idct(self, img_dct):
        """
        IDCT operation
        """
        batch_size, num_channels, height, width = img_dct.size()
        DCT_matrix = self.dct_matrix

        img_dct = img_dct.permute(0, 2, 3, 1)
        recover_img = torch.zeros_like(img_dct)

        for i in range(batch_size):
            for j in range(num_channels):
                img_channel = img_dct[i, :, :, j].reshape(height, -1)
                recover_img[i, :, :, j] = torch.matmul(DCT_matrix.T, torch.matmul(img_channel, DCT_matrix))

        recover_img = recover_img.permute(0, 3, 1, 2)
        recover_img = torch.clamp(recover_img, 0, 255).byte()
        return recover_img.float()
