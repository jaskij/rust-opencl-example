void kernel render_kernel(global uchar * rgba_out)
{
    const float3 final_colour_linear = (float3)(0.3f, 0.1f, 0.7f);

    // from http://chilliant.blogspot.jp/2012/08/srgb-approximations-for-hlsl.html
    const float3 final_colour_sRGB = min(255, max(0.0f, 1.055f * pow(final_colour_linear, 0.416666667f) - 0.055f) * 255);

    rgba_out[get_global_id(0) * 3 + 0] = (uchar)(final_colour_sRGB.x);
    rgba_out[get_global_id(0) * 3 + 1] = (uchar)(final_colour_sRGB.y);
    rgba_out[get_global_id(0) * 3 + 2] = (uchar)(final_colour_sRGB.z);
}
