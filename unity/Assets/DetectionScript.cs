using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UI = UnityEngine.UI;
using TMPro;

public class DetectionScript : MonoBehaviour
{

    public NNModel _model;
    public Texture2D _image;
    public UI.RawImage _imageView;
    public UI.RawImage webcamView;
    private WebCamTexture _webcamTexture;
    private IWorker _worker;

    public TMP_Text fpsText; // UI Text to display FPS
    public TMP_Text classText;

    private int _resizeLength = 640; // Length of one side of the square after resizing
    // Label information
    // There is a similar value in model.Metadata["names"], but it is registered as a JSON string
    // Define it separately as it does not seem to be parsable by the standard functionality.

    [Header("Detection Settings")]
    public float confidenceThreshold = 0.5f;
    public float iouThreshold = 0.75f;

    [Header("Preprocessing Settings")]
    public float brightnessFactor = 1.5f;
    private Dictionary<int, Color> colorMap = new Dictionary<int, Color>();

    private readonly string[] _labels = {
        "house", "bridge", "bus", "car", "church", "garden", "hospital", "palm tree",
        "police car", "river", "roller coaster", "school bus", "skyscraper", "tent",
        "The Eiffel Tower", "train", "tree"};

    private HashSet<string> classesToDetect = new HashSet<string> {
        "house", "bridge", "bus", "car", "church", "garden", "hospital", "palm tree",
        "police car", "river", "roller coaster", "school bus", "skyscraper", "tent",
        "The Eiffel Tower", "train", "tree"};// choose what class to detect 



    // Start is called before the first frame update
    void Start()
    {
        _webcamTexture = new WebCamTexture();
        webcamView.texture = _webcamTexture;
        _webcamTexture.Play();

    // load onnx model and create worker
        var model = ModelLoader.Load(_model);
        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

    }
    private float deltaTime = 0.0f;
    private float timeSinceLastUpdate = 0.0f;
    private float updateInterval = 0.2f;// Process frame every 0.2 seconds

    // Method to preprocess the texture
// Method to preprocess the texture with brightness enhancement
// Method to preprocess the texture with brightness enhancement
Texture2D PreprocessTexture(Texture2D originalTexture) {
    int width = originalTexture.width;
    int height = originalTexture.height;
    Texture2D processedTexture = new Texture2D(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Color pixelColor = originalTexture.GetPixel(x, y);
            // Enhance brightness
            Color brightenedColor = new Color(
                Mathf.Clamp01(pixelColor.r * brightnessFactor),
                Mathf.Clamp01(pixelColor.g * brightnessFactor),
                Mathf.Clamp01(pixelColor.b * brightnessFactor),
                1.0f);
            // Convert to grayscale, binarize, and invert colors as before...
            float grayScale = (brightenedColor.r + brightenedColor.g + brightenedColor.b) / 3f;
            grayScale = grayScale > 0.5f ? 1f : 0f;
            grayScale = 1f - grayScale;
            Color processedColor = new Color(grayScale, grayScale, grayScale, 1f);
            processedTexture.SetPixel(x, y, processedColor);
        }
    }
    processedTexture.Apply();
    return processedTexture;
}



    void Update() {
    timeSinceLastUpdate += Time.deltaTime;
    if (timeSinceLastUpdate >= updateInterval) {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        float fps = 1.0f / deltaTime;
        fpsText.text = $"FPS: {Mathf.Ceil(fps)}";
        timeSinceLastUpdate = 0.0f;
    }

    if (!_webcamTexture.didUpdateThisFrame) return;

    // Convert webcam feed to Texture2D
    Texture2D texture = new Texture2D(_webcamTexture.width, _webcamTexture.height);
    texture.SetPixels32(_webcamTexture.GetPixels32());
    texture.Apply();

    // Preprocess the texture
    Texture2D preprocessedTexture = PreprocessTexture(texture);

    // Resize the preprocessed texture for the model input
    Texture2D resizedTexture = ResizedTexture(preprocessedTexture, _resizeLength, _resizeLength);

    // Perform inference and other processing steps
    ProcessFrame(resizedTexture);

    // Clean up
    Destroy(texture);
    Destroy(preprocessedTexture);
}

    private void ProcessFrame(Texture2D texture)
    {
        // Convert to model input size and generate Tensor
        var resizedTexture = ResizedTexture(texture, _resizeLength, _resizeLength);
        Tensor inputTensor = new Tensor(resizedTexture, channels: 3);

        // Inference Execution
        _worker.Execute(inputTensor);

        // Analysis of results
        Tensor output0 = _worker.PeekOutput("output0");
        List<DetectionResult> ditects = ParseOutputs(output0, confidenceThreshold, iouThreshold);

        //_worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();
        // Visualize and display results
        VisualizeAndDisplayResults(ditects, texture);

        // Dispose of the resized texture
        Destroy(resizedTexture);
    }
    void OnDestroy()
    {
        if (_worker != null)
        {
            _worker.Dispose();
        }
    }
    private void VisualizeAndDisplayResults(List<DetectionResult> ditects, Texture2D originalTexture)
    {
        // Draw the result
        // Convert the result to the original size since we are analyzing a scaled-down image
        float scaleX = originalTexture.width / (float)_resizeLength;
        float scaleY = originalTexture.height / (float)_resizeLength;
        // clone image for displaying results
        Texture2D resultTexture = Instantiate(originalTexture);
        string detectedClasses = "";
        foreach (DetectionResult ditect in ditects)
        {
            detectedClasses += $"{_labels[ditect.classId]}: {ditect.score:0.00}\n";
            // Display Analysis Results
            Debug.Log($"{_labels[ditect.classId]}: {ditect.score:0.00}");
            Color color;

            if (!colorMap.ContainsKey(ditect.classId))
            {
                // Assign a random color if this class is not already in the colorMap
                color = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f));
                colorMap.Add(ditect.classId, color);
            }
            else
            {
                // Use the existing color from the colorMap
                color = colorMap[ditect.classId];
            }
                // Determine border size for rectangle (1 pixel border)
                int borderSize = 3;
            // Loop within the detected rectangle
            for (int x = (int)(ditect.x1 * scaleX); x < (int)(ditect.x2 * scaleX); x++)
            {
                for (int y = (int)(ditect.y1 * scaleY); y < (int)(ditect.y2 * scaleY); y++)
                {
                    // Check if the current pixel is on the edge of the rectangle and only then set the color
                    if (x < (int)(ditect.x1 * scaleX) + borderSize || x > (int)(ditect.x2 * scaleX) - borderSize ||
                        y < (int)(ditect.y1 * scaleY) + borderSize || y > (int)(ditect.y2 * scaleY) - borderSize)
                    {
                        // The detection result has the origin in the upper left corner, but Texture2D has the origin in the lower left corner, so the top and bottom are swapped.
                        resultTexture.SetPixel(x, resultTexture.height - y, color);
                    }
                }
            }
        }
        classText.text = detectedClasses;
        resultTexture.Apply();

        _imageView.texture = resultTexture;
    }
 
    private List<DetectionResult> ParseOutputs(Tensor output0, float threshold, float iouThres)
    {
        // Number of rows of detection results
        int outputWidth = output0.shape.width;

        // Candidate to be adopted as detection result
        List<DetectionResult> candidateDitects = new List<DetectionResult>();
        // Detection results to be used
        List<DetectionResult> ditects = new List<DetectionResult>();

        for (int i = 0; i < outputWidth; i++)
        {
            // Analyze detection results
            var result = new DetectionResult(output0, i);
            // Ignored if score is less than the specified value
            if (result.score < threshold || !classesToDetect.Contains(_labels[result.classId]))
            {
                continue;
            }
            // Add as candidate
            candidateDitects.Add(result);
        }

        // NonMaxSuppression processing
        // Adopt the one with the highest score in the overlapping rectangle
        while (candidateDitects.Count > 0)
        {
            int idx = 0;
            float maxScore = 0.0f;
            for (int i = 0; i < candidateDitects.Count; i++)
            {
                if (candidateDitects[i].score > maxScore)
                {
                    idx = i;
                    maxScore = candidateDitects[i].score;
                }
            }

            // get the result with the largest score and remove it from the list
            var cand = candidateDitects[idx];
            candidateDitects.RemoveAt(idx);

            // Add to results to be adopted
            ditects.Add(cand);

            List<int> deletes = new List<int>();
            for (int i = 0; i < candidateDitects.Count; i++)
            {
                // IOU Check
                float iou = Iou(cand, candidateDitects[i]);
                if (iou >= iouThres)
                {
                    deletes.Add(i);
                }
            }
            for (int i = deletes.Count - 1; i >= 0; i--)
            {
                candidateDitects.RemoveAt(deletes[i]);
            }

        }

        return ditects;

    }

    // Object overlap determination
    private float Iou(DetectionResult boxA, DetectionResult boxB)
    {
        if ((boxA.x1 == boxB.x1) && (boxA.x2 == boxB.x2) && (boxA.y1 == boxB.y1) && (boxA.y2 == boxB.y2))
        {
            return 1.0f;

        }
        else if (((boxA.x1 <= boxB.x1 && boxA.x2 > boxB.x1) || (boxA.x1 >= boxB.x1 && boxB.x2 > boxA.x1))
            && ((boxA.y1 <= boxB.y1 && boxA.y2 > boxB.y1) || (boxA.y1 >= boxB.y1 && boxB.y2 > boxA.y1)))
        {
            float intersection = (Mathf.Min(boxA.x2, boxB.x2) - Mathf.Max(boxA.x1, boxB.x1))
                * (Mathf.Min(boxA.y2, boxB.y2) - Mathf.Max(boxA.y1, boxB.y1));
            float union = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1) + (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1) - intersection;
            return (intersection / union);
        }

        return 0.0f;
    }



    // Image resizing process
    private static Texture2D ResizedTexture(Texture2D texture, int width, int height)
    {
        // RenderTexture
        var rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(texture, rt);
       
        var preRt = RenderTexture.active;
        RenderTexture.active = rt;
        var resizedTexture = new Texture2D(width, height);
        resizedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        resizedTexture.Apply();
        RenderTexture.active = preRt;
        RenderTexture.ReleaseTemporary(rt);
        return resizedTexture;
    }

}

// Detection result
class DetectionResult
{
    public float x1 { get; }
    public float y1 { get; }
    public float x2 { get; }
    public float y2 { get; }
    public int classId { get; }
    public float score { get; }

    public DetectionResult(Tensor t, int idx)
    {
        // The rectangle coordinates obtained from the detection result are 0:center x, 1:center y, 2:width, 3:height
        // Transform the coordinate system to be xy top-left xy bottom-right xy
        float halfWidth = t[0, 0, idx, 2] / 2;
        float halfHeight = t[0, 0, idx, 3] / 2;
        x1 = t[0, 0, idx, 0] - halfWidth;
        y1 = t[0, 0, idx, 1] - halfHeight;
        x2 = t[0, 0, idx, 0] + halfWidth;
        y2 = t[0, 0, idx, 1] + halfHeight;

        // Scores for each class are set in the remaining area
        // Maximum value is determined and set
        int classes = t.shape.channels - 4;
        score = 0f;
        for (int i = 0; i < classes; i++)
        {
            float classScore = t[0, 0, idx, i + 4];
            if (classScore < score)
            {
                continue;
            }
            classId = i;
            score = classScore;
        }
    }

}