import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 * 创建一个测试图片用于验证预处理
 */
public class CreateTestImage {
    public static void main(String[] args) throws Exception {
        // 创建一个简单的测试图片 (300x200)
        BufferedImage image = new BufferedImage(300, 200, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // 填充背景色
        g2d.setColor(Color.BLUE);
        g2d.fillRect(0, 0, 300, 200);
        
        // 画一些图形
        g2d.setColor(Color.RED);
        g2d.fillOval(50, 50, 100, 100);
        
        g2d.setColor(Color.GREEN);
        g2d.fillRect(150, 75, 100, 50);
        
        g2d.dispose();
        
        // 保存图片
        File outputFile = new File("test_image.jpg");
        ImageIO.write(image, "jpg", outputFile);
        
        System.out.println("测试图片已创建: " + outputFile.getAbsolutePath());
    }
}