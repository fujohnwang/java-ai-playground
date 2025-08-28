/**
 * 图像预处理的配置类 (扁平化版本)
 * <p>
 * 用于替代硬编码的参数，使预处理流程更加灵活。
 * 所有配置项均为类的直接字段，使用原始数据类型。
 */
public class ImageProcessorConfig {

    // --- 配置项字段 ---
    private boolean do_resize = true;
    private int shortest_edge = 256;

    /**
     * 插值算法
     * 2 -> 双线性插值 (Bilinear)
     * 3 -> 立方插值 (Cubic)
     * 0 -> 最近邻 (Nearest)
     */
    private int resample = 2;

    private boolean do_center_crop = true;
    private int crop_height = 224;
    private int crop_width = 224;

    private boolean do_rescale = true;
    private float rescale_factor = 1.0f / 255.0f;

    private boolean do_normalize = true;
    private float[] image_mean = {0.5f, 0.5f, 0.5f};
    private float[] image_std = {0.5f, 0.5f, 0.5f};

    // --- Getters and Setters ---

    public boolean isDo_resize() {
        return do_resize;
    }

    public void setDo_resize(boolean do_resize) {
        this.do_resize = do_resize;
    }

    public int getShortest_edge() {
        return shortest_edge;
    }

    public void setShortest_edge(int shortest_edge) {
        this.shortest_edge = shortest_edge;
    }

    public int getResample() {
        return resample;
    }

    public void setResample(int resample) {
        this.resample = resample;
    }

    public boolean isDo_center_crop() {
        return do_center_crop;
    }

    public void setDo_center_crop(boolean do_center_crop) {
        this.do_center_crop = do_center_crop;
    }

    public int getCrop_height() {
        return crop_height;
    }

    public void setCrop_height(int crop_height) {
        this.crop_height = crop_height;
    }

    public int getCrop_width() {
        return crop_width;
    }

    public void setCrop_width(int crop_width) {
        this.crop_width = crop_width;
    }

    /**
     * 便捷的setter方法，用于同时设置裁剪的高度和宽度。
     * @param height 裁剪高度
     * @param width  裁剪宽度
     */
    public void setCrop_size(int height, int width) {
        this.crop_height = height;
        this.crop_width = width;
    }

    public boolean isDo_rescale() {
        return do_rescale;
    }

    public void setDo_rescale(boolean do_rescale) {
        this.do_rescale = do_rescale;
    }

    public float getRescale_factor() {
        return rescale_factor;
    }

    public void setRescale_factor(float rescale_factor) {
        this.rescale_factor = rescale_factor;
    }

    public boolean isDo_normalize() {
        return do_normalize;
    }

    public void setDo_normalize(boolean do_normalize) {
        this.do_normalize = do_normalize;
    }

    public float[] getImage_mean() {
        return image_mean;
    }

    public void setImage_mean(float[] image_mean) {
        this.image_mean = image_mean;
    }

    public float[] getImage_std() {
        return image_std;
    }

    public void setImage_std(float[] image_std) {
        this.image_std = image_std;
    }
}