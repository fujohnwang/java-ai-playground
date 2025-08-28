import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * 从 ONNX 模型输出中提取 Embedding 的工具类。
 */
public final class ImageEmbeddingExtractor {

    // 私有构造函数，防止实例化
    private ImageEmbeddingExtractor() {}

    /**
     * 从 ONNX 推理结果中提取 embedding，并应用全局平均池化。
     * <p>
     * 此方法假定模型的输出是一个4D张量 (batch, channels, height, width)，
     * 并且 embedding 是通过对 height 和 width 维度进行平均池化得到的。
     *
     * @param result ONNX session 的运行结果
     * @return 经过全局平均池化后的一维 embedding 向量
     * @throws IllegalArgumentException 如果结果为空或格式不正确
     */
    public static float[] extractWithGlobalAvgPooling(OrtSession.Result result) {
        if (result == null) {
            throw new IllegalArgumentException("ONNX推理结果为空。");
        }

        try {
            // 通常我们关心的是第一个输出张量
            OnnxValue output = result.get(0);
            if (!(output instanceof OnnxTensor outputTensor)) {
                throw new IllegalArgumentException("模型输出不是一个有效的张量 (OnnxTensor)。");
            }

            Object outputValue = outputTensor.getValue();

            if (!(outputValue instanceof float[][][][] output4D)) {
                throw new IllegalArgumentException("模型输出张量的数据类型不是预期的 float[][][][]，而是 " + outputValue.getClass().getName());
            }

            // 验证维度，至少需要4维且不为空
            if (output4D.length == 0 || output4D[0].length == 0 || output4D[0][0].length == 0 || output4D[0][0][0].length == 0) {
                throw new IllegalArgumentException("输出张量的维度为空。");
            }

            // 执行全局平均池化
            int channels = output4D[0].length;
            int height = output4D[0][0].length;
            int width = output4D[0][0][0].length;

            float[] embedding = new float[channels];
            for (int c = 0; c < channels; c++) {
                float channelSum = 0;
                // 我们只处理batch中的第一个元素 (index 0)
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        channelSum += output4D[0][c][h][w];
                    }
                }
                embedding[c] = channelSum / (height * width);
            }

            return embedding;

        } catch (OrtException e) {
            // 将检查型异常转换为运行时异常，简化调用方代码
            throw new RuntimeException("提取ONNX结果时发生错误", e);
        }
    }
}
