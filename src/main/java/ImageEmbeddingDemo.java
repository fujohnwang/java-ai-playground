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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

public class ImageEmbeddingDemo implements AutoCloseable {
    private OrtEnvironment environment;
    private OrtSession session;

    /**
     * Constructs an ImageEmbedding instance.
     *
     * @param modelPath The file path to the ONNX model.
     * @throws OrtException If there is an error initializing the ONNX Runtime environment or session.
     */
    public ImageEmbeddingDemo(String modelPath) throws OrtException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath, new OrtSession.SessionOptions());
        System.out.println("ONNX model loaded from: " + modelPath);
        System.out.println("Input names: " + session.getInputNames());
        System.out.println("Output names: " + session.getOutputNames());
    }

    /**
     * Generates an embedding for the given image file path.
     *
     * @param imagePath The path to the image file.
     * @return A float array representing the image embedding.
     * @throws IOException  If the image file cannot be read.
     * @throws OrtException If there is an error during model inference.
     */
    public float[] embed(String imagePath) throws IOException, OrtException {
        // 1. Preprocess the image using the utility class
        // Assuming the utility returns a float array in the shape [1, C, H, W]
        float[] preprocessedData = ImagePreprocessingUtils.preprocessImage(imagePath);

        // 2. Create an ONNX tensor from the preprocessed data
        // Assuming the model's input node name is "input"
        // 创建输入张量
        long[] inputShape = {1, 3, 224, 224};
        OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(preprocessedData), inputShape);

        // 执行推理
        System.out.println("执行模型推理...");
        try (OrtSession.Result result = session.run(Collections.singletonMap("pixel_values", inputTensor))) {
            // 4. Extract the embedding from the result
            return ImageEmbeddingExtractor.extractWithGlobalAvgPooling(result);
        }
    }


    @Override
    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (environment != null) {
            environment.close();
        }
    }

    public static void main(String[] args) {
        String modelPath = "/Users/fq/workspace.python/optimum-runner/onnx/mobilenet-v2/model.onnx";
        try (ImageEmbeddingDemo imageEmbedding = new ImageEmbeddingDemo(modelPath)) {
            // Generate the embedding
//            System.out.println("warm up...");
//            for (var i = 0; i < 100; i++) {
//                imageEmbedding.embed("/Users/fq/福匠素材库.library/images/MESGARN2V4E0W.info/Output for the Skill Character Mix.png");
//            }
//
//            System.out.println("start embed...");
//            long start = System.currentTimeMillis();
//            for (var i = 0; i < 10000; i++) {
//                imageEmbedding.embed("/Users/fq/福匠素材库.library/images/MESGARN2V4E0W.info/Output for the Skill Character Mix.png");
//            }
//            long end = System.currentTimeMillis();
//            System.out.printf("it takes %s milli to calculate 10000 embeddings of image, avg=%s ms/image", String.valueOf(end - start), String.valueOf((end - start) / 10000));
//            // it takes 447455 milli to calculate 10000 embeddings of image, avg=44 ms/image执行模型推理...

//            float[] embedding = imageEmbedding.embed("/Users/fq/福匠素材库.library/images/MESGARN2V4E0W.info/Output for the Skill Character Mix.png");
//            float[] embedding = imageEmbedding.embed("/Users/fq/福匠素材库.library/images/MESGBS2F1QQ1X.info/小黄人 minion joker.png");

            float[] embedding = imageEmbedding.embed("/Users/fq/福匠素材库.library/images/MATN4MON5564I.info/Clipboard - 2025-05-18 20.36.39.png");


            System.out.println("Successfully generated embedding.");
            System.out.println("Embedding dimension: " + embedding.length);
            System.out.println("Embedding preview: " + Arrays.toString(embedding) + " ");

        } catch (OrtException e) {
            System.err.println("ONNX Runtime error: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("An unexpected error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
