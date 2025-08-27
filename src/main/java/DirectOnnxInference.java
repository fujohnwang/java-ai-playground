import java.nio.file.Paths;
import java.util.Arrays;

/**
 * 简化版本：直接使用我们的预处理工具，然后尝试最简单的DJL方式
 */
public class DirectOnnxInference {

    public static void main(String[] args) throws Exception {
        // 模型和图像路径 - 请修改为实际存在的路径
        String modelPath = "/Users/fq/workspace.python/optimum-runner/onnx/mobilenet-v2/model.onnx";
        String imagePath = "test_image.jpg"; // 请替换为实际存在的图片路径

        // 检查文件是否存在
        if (!java.nio.file.Files.exists(Paths.get(imagePath))) {
            System.err.println("错误：图片文件不存在: " + imagePath);
            return;
        }

        // 使用我们的工具类进行预处理
        System.out.println("开始图像预处理...");
        float[] preprocessedData = ImagePreprocessingUtils.preprocessImage(imagePath);
        System.out.println("预处理完成，数据长度: " + preprocessedData.length);
        System.out.println("预处理后的数据形状应该是: [1, 3, 224, 224] = " + (1 * 3 * 224 * 224));
        
        // 验证数据
        System.out.println("前5个像素值: " + Arrays.toString(Arrays.copyOf(preprocessedData, 5)));
        System.out.println("数据范围检查 - 最小值: " + getMin(preprocessedData) + ", 最大值: " + getMax(preprocessedData));
        
        System.out.println("\n图像预处理成功完成！");
        System.out.println("现在你可以使用这个 preprocessedData 数组来创建 NDArray 进行推理。");
        System.out.println("建议使用更底层的 ONNX Runtime Java API 而不是 DJL 的 Predictor。");
    }
    
    private static float getMin(float[] array) {
        float min = Float.MAX_VALUE;
        for (float f : array) {
            if (f < min) min = f;
        }
        return min;
    }
    
    private static float getMax(float[] array) {
        float max = Float.MIN_VALUE;
        for (float f : array) {
            if (f > max) max = f;
        }
        return max;
    }
}