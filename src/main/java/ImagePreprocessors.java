import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * 图像预处理工具类（重构版本）
 * <p>
 * 该类将预处理的逻辑与配置分离。所有的处理步骤均由一个
 * {@link ImageProcessorConfig} 对象来驱动，从而实现灵活可配的预处理流程。
 * <p>
 * 所有方法均为静态方法，方便直接调用。
 */
public final class ImagePreprocessors {

    // 私有构造函数，防止实例化
    private ImagePreprocessors() {}

    /**
     * 主入口方法：根据指定的配置对图像进行完整的预处理。
     *
     * @param image  待处理的 {@link BufferedImage} 对象。
     * @param config 包含所有预处理参数的 {@link ImageProcessorConfig} 对象。
     * @return 预处理后的一维 float 数组，可直接用于创建 NDArray/Tensor。
     */
    public static float[] preprocessImage(BufferedImage image, ImageProcessorConfig config) {
        // 确保输入不为空
        if (image == null) {
            throw new IllegalArgumentException("输入图像不能为 null");
        }
        if (config == null) {
            throw new IllegalArgumentException("配置对象不能为 null");
        }

        // 1. 调整尺寸 (Resize)
        BufferedImage currentImage = image;
        if (config.isDo_resize()) {
            currentImage = resizeByShortestEdge(currentImage, config.getShortest_edge(), config.getResample());
        }

        // 2. 中心裁剪 (Center Crop)
        if (config.isDo_center_crop()) {
            currentImage = centerCrop(currentImage, config.getCrop_width(), config.getCrop_height());
        }

        // 3. 提取RGB像素
        int[][][] rgbPixels = extractRGBPixels(currentImage);

        // 4. 像素值重缩放 (Rescale)
        float[][][] rescaled = new float[rgbPixels.length][rgbPixels[0].length][3];
        if (config.isDo_rescale()) {
            rescaled = rescalePixels(rgbPixels, config.getRescale_factor());
        } else {
            // 如果不进行rescale，也需要将int转换为float
            for (int y = 0; y < rgbPixels.length; y++) {
                for (int x = 0; x < rgbPixels[0].length; x++) {
                    for (int c = 0; c < 3; c++) {
                        rescaled[y][x][c] = rgbPixels[y][x][c];
                    }
                }
            }
        }

        // 5. 标准化 (Normalize)
        float[][][] normalized = rescaled;
        if (config.isDo_normalize()) {
            normalized = normalizePixels(rescaled, config.getImage_mean(), config.getImage_std());
        }

        // 6. 转换为NCHW格式
        float[][][][] nchw = convertToNCHW(normalized);

        // 7. 展平为一维数组
        return flattenNCHW(nchw);
    }

    /**
     * 便捷的重载方法：从文件路径加载图像并进行预处理。
     *
     * @param imagePath 图像文件的路径。
     * @param config    包含所有预处理参数的 {@link ImageProcessorConfig} 对象。
     * @return 预处理后的一维 float 数组。
     * @throws IOException 如果图像文件读取失败。
     */
    public static float[] preprocessImage(String imagePath, ImageProcessorConfig config) throws IOException {
        BufferedImage image = loadImage(imagePath);
        return preprocessImage(image, config);
    }

    // --- 以下为私有的辅助方法，逻辑与原 ImagePreprocessingUtils 保持一致 ---

    private static BufferedImage loadImage(String imagePath) throws IOException {
        File f = new File(imagePath);
        if (!f.exists()) {
            throw new IOException("图像文件不存在: " + imagePath);
        }
        return ImageIO.read(f);
    }

    private static BufferedImage resizeByShortestEdge(BufferedImage image, int shortestEdge, int resample) {
        int width = image.getWidth();
        int height = image.getHeight();

        float scale = (height < width) ? (float) shortestEdge / height : (float) shortestEdge / width;
        int newWidth = Math.round(width * scale);
        int newHeight = Math.round(height * scale);

        Object interpolation;
        switch (resample) {
            case 3:
                interpolation = RenderingHints.VALUE_INTERPOLATION_BICUBIC;
                break;
            case 0:
                interpolation = RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR;
                break;
            case 2:
            default: // 默认为双线性插值
                interpolation = RenderingHints.VALUE_INTERPOLATION_BILINEAR;
                break;
        }

        BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolation);
        g2d.drawImage(image, 0, 0, newWidth, newHeight, null);
        g2d.dispose();
        return resized;
    }

    private static BufferedImage centerCrop(BufferedImage image, int cropWidth, int cropHeight) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width < cropWidth || height < cropHeight) {
            throw new IllegalArgumentException(
                String.format("输入图像尺寸 (%d, %d) 小于裁剪尺寸 (%d, %d)，无法裁剪", width, height, cropWidth, cropHeight)
            );
        }
        int startX = (width - cropWidth) / 2;
        int startY = (height - cropHeight) / 2;
        return image.getSubimage(startX, startY, cropWidth, cropHeight);
    }

    private static int[][][] extractRGBPixels(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[][][] pixels = new int[height][width][3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                pixels[y][x][0] = (rgb >> 16) & 0xFF; // Red
                pixels[y][x][1] = (rgb >> 8) & 0xFF;  // Green
                pixels[y][x][2] = rgb & 0xFF;         // Blue
            }
        }
        return pixels;
    }

    private static float[][][] rescalePixels(int[][][] pixels, float rescaleFactor) {
        int height = pixels.length;
        int width = pixels[0].length;
        float[][][] rescaled = new float[height][width][3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    rescaled[y][x][c] = pixels[y][x][c] * rescaleFactor;
                }
            }
        }
        return rescaled;
    }

    private static float[][][] normalizePixels(float[][][] pixels, float[] mean, float[] std) {
        int height = pixels.length;
        int width = pixels[0].length;
        float[][][] normalized = new float[height][width][3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    normalized[y][x][c] = (pixels[y][x][c] - mean[c]) / std[c];
                }
            }
        }
        return normalized;
    }

    private static float[][][][] convertToNCHW(float[][][] pixels) {
        int height = pixels.length;
        int width = pixels[0].length;
        float[][][][] nchw = new float[1][3][height][width];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    nchw[0][c][y][x] = pixels[y][x][c];
                }
            }
        }
        return nchw;
    }

    private static float[] flattenNCHW(float[][][][] nchw) {
        int batch = nchw.length;
        int channels = nchw[0].length;
        int height = nchw[0][0].length;
        int width = nchw[0][0][0].length;
        float[] flattened = new float[batch * channels * height * width];
        int index = 0;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        flattened[index++] = nchw[b][c][h][w];
                    }
                }
            }
        }
        return flattened;
    }
}