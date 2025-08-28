import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
 * 使用原生ONNX Runtime Java API进行推理
 */
public class NativeOnnxInference {

    public static void main(String[] args) throws Exception {
        // 模型和图像路径
        String modelPath = "/Users/fq/workspace.python/optimum-runner/onnx/mobilenet-v2/model.onnx";
        String imagePath = "test_image.jpg";

        // 检查文件是否存在
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(imagePath))) {
            System.err.println("错误：图片文件不存在: " + imagePath);
            return;
        }

        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelPath))) {
            System.err.println("错误：模型文件不存在: " + modelPath);
            System.out.println("请修改 modelPath 为实际的ONNX模型路径");
            return;
        }

        // 使用我们的工具类进行预处理
        System.out.println("开始图像预处理...");
        float[] preprocessedData = ImagePreprocessingUtils.preprocessImage(imagePath);
        System.out.println("预处理完成，数据长度: " + preprocessedData.length);
        
        // 检查预处理后的输入数据
        float inputMin = Float.MAX_VALUE;
        float inputMax = Float.MIN_VALUE;
        float inputSum = 0;
        for (float value : preprocessedData) {
            if (value < inputMin) inputMin = value;
            if (value > inputMax) inputMax = value;
            inputSum += value;
        }
        float inputMean = inputSum / preprocessedData.length;
        
        System.out.println("=== 输入数据统计 ===");
        System.out.println("输入最小值: " + inputMin);
        System.out.println("输入最大值: " + inputMax);
        System.out.println("输入平均值: " + inputMean);
        System.out.println("前10个输入值: " + Arrays.toString(Arrays.copyOf(preprocessedData, Math.min(10, preprocessedData.length))));

        // 使用原生ONNX Runtime API
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

            // 获取模型输入输出信息
            System.out.println("模型输入信息:");
            for (NodeInfo inputInfo : session.getInputInfo().values()) {
                System.out.println("  输入名称: " + inputInfo.getName());
                System.out.println("  输入形状: " + Arrays.toString(((TensorInfo) inputInfo.getInfo()).getShape()));
                System.out.println("  输入类型: " + ((TensorInfo) inputInfo.getInfo()).type);
            }

            System.out.println("模型输出信息:");
            for (NodeInfo outputInfo : session.getOutputInfo().values()) {
                System.out.println("  输出名称: " + outputInfo.getName());
                System.out.println("  输出形状: " + Arrays.toString(((TensorInfo) outputInfo.getInfo()).getShape()));
                System.out.println("  输出类型: " + ((TensorInfo) outputInfo.getInfo()).type);
            }

            // 创建输入张量
            long[] inputShape = {1, 3, 224, 224};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(preprocessedData), inputShape);

            // 执行推理
            System.out.println("执行模型推理...");
            try (OrtSession.Result result = session.run(Collections.singletonMap("pixel_values", inputTensor))) {
                
                System.out.println("推理完成！");
                System.out.println("输出张量数量: " + result.size());
                
                // 获取输出 - 使用正确的API
                for (int i = 0; i < result.size(); i++) {
                    OnnxValue output = result.get(i);
                    if (output instanceof OnnxTensor) {
                        OnnxTensor outputTensor = (OnnxTensor) output;
                        System.out.println("输出张量 " + i + " 形状: " + Arrays.toString(outputTensor.getInfo().getShape()));
                        
                        // 获取输出数据 - 输出是4维数组 [1, 1280, 7, 7]
                        Object outputValue = outputTensor.getValue();
                        if (outputValue instanceof float[][][][]) {
                            float[][][][] output4D = (float[][][][]) outputValue;
                            System.out.println("输出数据维度: [" + output4D.length + ", " + output4D[0].length + ", " + output4D[0][0].length + ", " + output4D[0][0][0].length + "]");
                            
                            // 展平为1维数组以便查看
                            int totalSize = output4D.length * output4D[0].length * output4D[0][0].length * output4D[0][0][0].length;
                            float[] flattened = new float[totalSize];
                            int index = 0;
                            for (int b = 0; b < output4D.length; b++) {
                                for (int c = 0; c < output4D[0].length; c++) {
                                    for (int h = 0; h < output4D[0][0].length; h++) {
                                        for (int w = 0; w < output4D[0][0][0].length; w++) {
                                            flattened[index++] = output4D[b][c][h][w];
                                        }
                                    }
                                }
                            }
                            
                            System.out.println("展平后数据长度: " + flattened.length);
                            
                            // 打印更多统计信息
                            float min = Float.MAX_VALUE;
                            float max = Float.MIN_VALUE;
                            float sum = 0;
                            int nonZeroCount = 0;
                            
                            for (float value : flattened) {
                                if (value < min) min = value;
                                if (value > max) max = value;
                                sum += value;
                                if (value != 0) nonZeroCount++;
                            }
                            
                            float mean = sum / flattened.length;
                            
                            System.out.println("=== 输出统计信息 ===");
                            System.out.println("最小值: " + min);
                            System.out.println("最大值: " + max);
                            System.out.println("平均值: " + mean);
                            System.out.println("非零值数量: " + nonZeroCount + " / " + flattened.length);
                            System.out.println("零值比例: " + String.format("%%.2f%%", (flattened.length - nonZeroCount) * 100.0 / flattened.length));
                            
                            // 打印前20个值
                            System.out.println("前20个输出值: " + Arrays.toString(Arrays.copyOf(flattened, Math.min(20, flattened.length))));
                            
                            // 打印中间20个值
                            int midStart = flattened.length / 2 - 10;
                            System.out.println("中间20个输出值 (从索引" + midStart + "开始): " + 
                                Arrays.toString(Arrays.copyOfRange(flattened, midStart, Math.min(midStart + 20, flattened.length))));
                            
                            // 打印最后20个值
                            int endStart = Math.max(0, flattened.length - 20);
                            System.out.println("最后20个输出值: " + 
                                Arrays.toString(Arrays.copyOfRange(flattened, endStart, flattened.length)));
                            
                            // 如果有非零值，找到第一个非零值的位置
                            if (nonZeroCount > 0) {
                                for (int idx = 0; idx < flattened.length; idx++) {
                                    if (flattened[idx] != 0) {
                                        System.out.println("第一个非零值在索引 " + idx + ": " + flattened[idx]);
                                        break;
                                    }
                                }
                            }
                            
                            System.out.println("\n=== 不同的Embedding提取方法 ===");
                            
                            // --- 调用外部工具类进行全局平均池化 ---
                            float[] globalAvgPooling = ImageEmbeddingExtractor.extractWithGlobalAvgPooling(result);

                            // 方法2: 全局最大池化 (保留这里的实现用于对比)
                            float[] globalMaxPooling = new float[1280];
                            for (int c = 0; c < 1280; c++) {
                                float channelMax = Float.MIN_VALUE;
                                for (int h = 0; h < 7; h++) {
                                    for (int w = 0; w < 7; w++) {
                                        if (output4D[0][c][h][w] > channelMax) {
                                            channelMax = output4D[0][c][h][w];
                                        }
                                    }
                                }
                                globalMaxPooling[c] = channelMax;
                            }
                            
                            // 方法3: 取中心点 (3,3)
                            float[] centerPoint = new float[1280];
                            for (int c = 0; c < 1280; c++) {
                                centerPoint[c] = output4D[0][c][3][3]; // 中心点
                            }
                            
                            // 统计不同方法的结果
                            System.out.println("1. 全局平均池化 (1280维) - [来自 ImageEmbeddingExtractor]:");
                            printEmbeddingStats("   GAP", globalAvgPooling);
                            
                            System.out.println("2. 全局最大池化 (1280维):");
                            printEmbeddingStats("   GMP", globalMaxPooling);
                            
                            System.out.println("3. 中心点特征 (1280维):");
                            printEmbeddingStats("   Center", centerPoint);
                            
                            System.out.println("4. 展平所有值 (62720维):");
                            printEmbeddingStats("   Flatten", flattened);
                        } else {
                            System.out.println("输出数据类型: " + outputValue.getClass().getName());
                        }
                        
                        System.out.println("--- 成功获取图像embedding! ---");
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("推理过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 打印embedding向量的统计信息
     */
    private static void printEmbeddingStats(String name, float[] embedding) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        float sum = 0;
        int nonZeroCount = 0;
        
        for (float value : embedding) {
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
            if (value != 0) nonZeroCount++;
        }
        
        float mean = sum / embedding.length;
        
        System.out.println(name + " - 维度: " + embedding.length + 
                          ", 范围: [" + String.format("%%.4f", min) + ", " + String.format("%%.4f", max) + "]" + 
                          ", 均值: " + String.format("%%.4f", mean) + 
                          ", 非零: " + nonZeroCount + "/" + embedding.length);
        System.out.println(name + " - 前5个值: " + Arrays.toString(Arrays.copyOf(embedding, Math.min(5, embedding.length))));
    }
}
