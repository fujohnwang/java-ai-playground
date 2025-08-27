/**
 * <pre>
 * :::    ::: :::::::::: :::::::::: :::     :::  ::::::::  :::
 * :+:   :+:  :+:        :+:        :+:     :+: :+:    :+: :+:
 * +:+  +:+   +:+        +:+        +:+     +:+ +:+    +:+ +:+
 * +#++:++    +#++:++#   +#++:++#   +#+     +:+ +#+    +:+ +#+
 * +#+  +#+   +#+        +#+         +#+   +#+  +#+    +#+ +#+
 * #+#   #+#  #+#        #+#          #+#+#+#   #+#    #+# #+#
 *  * </pre>
 * <p>
 * KEEp eVOLution!
 * <p>
 *
 * @author fq@keevol.cn
 * @since 2017.5.12
 * <p>
 * Copyright 2017 © 杭州福强科技有限公司版权所有 (<a href="https://www.keevol.cn">keevol.cn</a>)
 */





import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class DjlCorrectPreprocessing {

    // 使用 Netron 确认您的模型输入输出的真实名称
    private static final String INPUT_TENSOR_NAME = "pixel_values";
    private static final String OUTPUT_TENSOR_NAME = "last_hidden_state";

    public static void main(String[] args) throws Exception {
        // --- 请务必修改为一张真实存在的图片路径 ---
        Path modelDir = Paths.get("/Users/fq/workspace.python/optimum-runner/onnx/mobilenet-v2/model.onnx"); // 指向包含所有文件的目录
        Path imagePath = Paths.get("/Users/fq/福匠素材库.library/images/MEBI1RRCSSF1A.info/panda.png");
        // ------------------------------------------

        if (!Files.exists(imagePath)) {
            System.err.println("错误：图片文件不存在: " + imagePath);
            return;
        }

        Translator<Image, float[]> translator = new MobileNetTranslator();

        Criteria<Image, float[]> criteria = Criteria.builder()
                .setTypes(Image.class, float[].class)
                .optModelPath(modelDir)
                .optEngine("OnnxRuntime")
                .optTranslator(translator)
                .build();

        try (ZooModel<Image, float[]> model = criteria.loadModel();
             Predictor<Image, float[]> predictor = model.newPredictor()) {

            Image image = ImageFactory.getInstance().fromFile(imagePath);
            float[] embedding = predictor.predict(image);

            System.out.println("--- 成功 ---");
            System.out.println("Embedding 维度: " + embedding.length);
            System.out.println("前5个维度的值: " + Arrays.toString(Arrays.copyOf(embedding, 5)));
        }
    }

    /**
     * 正确的自定义 Translator 实现，严格按照 preprocessor_config.json 配置
     */
    private static final class MobileNetTranslator implements Translator<Image, float[]> {

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            try {
                NDManager manager = ctx.getNDManager();
                
                // 将DJL Image保存为临时文件，然后使用我们的工具类处理
                java.nio.file.Path tempFile = Paths.get("/Users/fq/福匠素材库.library/images/MEBI1RRCSSF1A.info/panda.png");;
                input.save(java.nio.file.Files.newOutputStream(tempFile), "png");
                
                // 使用纯Java工具类进行预处理
                float[] processedData = ImagePreprocessingUtils.preprocessImage(tempFile.toString());
                
                // 删除临时文件
                java.nio.file.Files.delete(tempFile);
                
                // 创建NDArray，直接指定形状 [1, 3, 224, 224]
                NDArray array = manager.create(processedData, new ai.djl.ndarray.types.Shape(1, 3, 224, 224));
                
                // 设置输入张量名称
                array.setName(INPUT_TENSOR_NAME);
                
                return new NDList(array);
                
            } catch (Exception e) {
                throw new RuntimeException("图像预处理失败", e);
            }
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            // 尝试按名称获取输出张量
            NDArray embedding = list.get(OUTPUT_TENSOR_NAME);
            
            // 如果按名称找不到，尝试获取第一个输出
            if (embedding == null && list.size() > 0) {
                embedding = list.get(0);
                System.out.println("警告: 未找到名为 '" + OUTPUT_TENSOR_NAME + "' 的输出张量，使用第一个输出");
            }
            
            if (embedding == null) {
                throw new IllegalStateException("模型没有输出任何张量");
            }
            
            // 如果输出有多个维度，可能需要展平或选择特定维度
            // 对于 MobileNetV2，通常输出是 (batch_size, sequence_length, hidden_size) 或类似格式
            long[] shape = embedding.getShape().getShape();
            System.out.println("输出张量形状: " + Arrays.toString(shape));
            
            return embedding.toFloatArray();
        }
    }
}