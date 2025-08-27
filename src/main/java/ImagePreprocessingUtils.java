import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * 图像预处理工具类
 * 使用纯Java实现，不依赖DJL库，按照HuggingFace preprocessor_config.json配置进行预处理
 */
public class ImagePreprocessingUtils {

    /**
     * 从文件加载图像
     */
    public static BufferedImage loadImage(String imagePath) throws IOException {
        return ImageIO.read(new File(imagePath));
    }

    /**
     * 步骤1: do_resize - 按最短边缩放到指定尺寸，保持宽高比
     * @param image 原始图像
     * @param shortestEdge 最短边目标尺寸 (如256)
     * @return 缩放后的图像
     */
    public static BufferedImage resizeByShortestEdge(BufferedImage image, int shortestEdge) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // 计算缩放比例
        float scale;
        if (height < width) {
            scale = (float) shortestEdge / height;
        } else {
            scale = (float) shortestEdge / width;
        }
        
        int newWidth = Math.round(width * scale);
        int newHeight = Math.round(height * scale);
        
        // 使用双线性插值进行缩放
        BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(image, 0, 0, newWidth, newHeight, null);
        g2d.dispose();
        
        return resized;
    }

    /**
     * 步骤2: do_center_crop - 中心裁剪到指定尺寸
     * @param image 输入图像
     * @param cropWidth 裁剪宽度
     * @param cropHeight 裁剪高度
     * @return 裁剪后的图像
     */
    public static BufferedImage centerCrop(BufferedImage image, int cropWidth, int cropHeight) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        int startX = (width - cropWidth) / 2;
        int startY = (height - cropHeight) / 2;
        
        return image.getSubimage(startX, startY, cropWidth, cropHeight);
    }    /**
  
   * 步骤3: 提取RGB像素数据
     * @param image 输入图像
     * @return RGB像素数组 [height][width][channels] 格式，值范围0-255
     */
    public static int[][][] extractRGBPixels(BufferedImage image) {
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

    /**
     * 步骤4: do_rescale - 像素值缩放
     * @param pixels RGB像素数组
     * @param rescaleFactor 缩放因子 (如 1/255 = 0.00392156862745098)
     * @return 缩放后的像素数组，值范围0-1
     */
    public static float[][][] rescalePixels(int[][][] pixels, float rescaleFactor) {
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

    /**
     * 步骤5: do_normalize - 标准化
     * @param pixels 输入像素数组
     * @param mean 均值数组 [R, G, B]
     * @param std 标准差数组 [R, G, B]
     * @return 标准化后的像素数组
     */
    public static float[][][] normalizePixels(float[][][] pixels, float[] mean, float[] std) {
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
 /**
     * 步骤6: 转换为NCHW格式
     * @param pixels HWC格式的像素数组 [height][width][channels]
     * @return NCHW格式的数组 [batch][channels][height][width]，batch=1
     */
    public static float[][][][] convertToNCHW(float[][][] pixels) {
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

    /**
     * 转换为一维数组，用于创建NDArray
     * @param nchw NCHW格式的4维数组
     * @return 一维float数组
     */
    public static float[] flattenNCHW(float[][][][] nchw) {
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

    /**
     * 完整的预处理流程
     * @param imagePath 图像文件路径
     * @return 预处理后的一维float数组，可直接用于创建NDArray
     */
    public static float[] preprocessImage(String imagePath) throws IOException {
        // 加载图像
        BufferedImage image = loadImage(imagePath);
        
        // 1. 按最短边缩放到256
        BufferedImage resized = resizeByShortestEdge(image, 256);
        
        // 2. 中心裁剪到224x224
        BufferedImage cropped = centerCrop(resized, 224, 224);
        
        // 3. 提取RGB像素
        int[][][] rgbPixels = extractRGBPixels(cropped);
        
        // 4. 缩放像素值 (0-255 -> 0-1)
        float[][][] rescaled = rescalePixels(rgbPixels, 1.0f / 255.0f);
        
        // 5. 标准化 (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        float[] mean = {0.5f, 0.5f, 0.5f};
        float[] std = {0.5f, 0.5f, 0.5f};
        float[][][] normalized = normalizePixels(rescaled, mean, std);
        
        // 6. 转换为NCHW格式
        float[][][][] nchw = convertToNCHW(normalized);
        
        // 7. 展平为一维数组
        return flattenNCHW(nchw);
    }
}