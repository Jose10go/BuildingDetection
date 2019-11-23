namespace CNTKUtil
{
    public interface IData
    {
        float[] Features { get; }
        float[] Labels { get; }
    }
}