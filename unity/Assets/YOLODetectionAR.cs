using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;
using TMPro;

public class YOLODetectionAR : MonoBehaviour
{
    public NNModel modelAsset;
    public GameObject labelPrefab; // Prefab with Text component for displaying class names
    public Transform labelsParent; // UI parent to keep the scene organized
    public RawImage webcamDisplay; // Assign this in the Inspector

    private Model runtimeModel;
    private IWorker worker;
    private WebCamTexture webcamTexture;

    private Dictionary<int, string> classLabels = new Dictionary<int, string>
    {
        {0, "house"}, {1, "bridge"}, {2, "bus"}, {3, "car"}, {4, "church"},
        {5, "garden"}, {6, "hospital"}, {7, "palm tree"}, {8, "police car"},
        {9, "river"}, {10, "roller coaster"}, {11, "school bus"}, {12, "skyscraper"},
        {13, "tent"}, {14, "The Eiffel Tower"}, {15, "train"}, {16, "tree"}
    };

    void Start()
{
    runtimeModel = ModelLoader.Load(modelAsset);
    if (runtimeModel != null)
    {
        Debug.Log("Model loaded successfully.");
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
    }
    else
    {
        Debug.LogError("Failed to load model.");
    }

    webcamTexture = new WebCamTexture();
    if (webcamDisplay != null) {
        webcamDisplay.texture = webcamTexture;
    } else {
        Debug.LogError("WebcamDisplay RawImage has not been assigned in the inspector.");
    }
    webcamTexture.Play();
}


    void Update()
    {
        if (webcamTexture.didUpdateThisFrame)
        {
            ProcessWebcamInput();
        }
    }

    void ProcessWebcamInput()
    {
        Texture2D inputImage = new Texture2D(webcamTexture.width, webcamTexture.height);
        inputImage.SetPixels(webcamTexture.GetPixels());
        inputImage.Apply();

        Tensor inputTensor = Preprocess(inputImage);
        worker.Execute(inputTensor);
        Tensor outputTensor = worker.PeekOutput();

        ProcessOutput(outputTensor);

        inputTensor.Dispose();
        outputTensor.Dispose();
        Destroy(inputImage);
    }

    private Tensor Preprocess(Texture2D inputImage)
    {
        Texture2D resizedImage = Resize(inputImage, 640, 640);
        return new Tensor(resizedImage, 3);
    }

    private Texture2D Resize(Texture2D texture2D, int targetX, int targetY)
    {
        RenderTexture rt = new RenderTexture(targetX, targetY, 24);
        Graphics.Blit(texture2D, rt);
        Texture2D result = new Texture2D(targetX, targetY, texture2D.format, false);
        RenderTexture.active = rt;
        result.ReadPixels(new Rect(0, 0, targetX, targetY), 0, 0);
        result.Apply();
        RenderTexture.active = null;
        rt.Release();
        return result;
    }

   private void ProcessOutput(Tensor outputTensor)
{
    bool detected = false;

    // Assuming the output format [1, 1, 8400, 21] means we have 400 detections, each represented by 21 values
    int detections = outputTensor.shape[2] / 21; // Calculate the number of detections

    for (int i = 0; i < detections; i++)
    {
        float confidence = outputTensor[0, 0, i * 21, 0]; // Assuming first value of each detection is confidence

        if (confidence <= 0.2) // Threshold to filter out low confidence detections
            continue;

        int classId = -1;
        float maxProb = 0f;

        for (int j = 1; j < 21; j++) // Start from 1 assuming 0 is confidence
        {
            float prob = outputTensor[0, 0, i * 21 + j, 0];
            if (prob > maxProb)
            {
                maxProb = prob;
                classId = j - 1; // Adjust because class labels start from 0
            }
        }

        // Check if a valid class ID was found
        if (classId != -1)
        {
            detected = true;
            string detectedClassName = classLabels.ContainsKey(classId) ? classLabels[classId] : "Unknown";
            Debug.Log($"Detected: {detectedClassName} with confidence: {confidence} and probability: {maxProb}");
            DisplayClassName(detectedClassName);
            break; // For simplicity, assuming only one detection is handled
        }
    }

   
}



    private void DisplayClassName(string className)
    {
        foreach (Transform child in labelsParent)
        {
            Destroy(child.gameObject);
        }

        GameObject labelGo = Instantiate(labelPrefab, labelsParent);
        TextMeshProUGUI labelText = labelGo.GetComponentInChildren<TextMeshProUGUI>();
        if (labelText == null)
        {
            Debug.LogError("No TextMeshProUGUI component found on labelPrefab or its children.");
            return;
        }
        labelText.text = className;
    }
}
