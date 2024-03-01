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
    public Texture2D testImage; // Assign this in the Inspector with your static image

    private Model runtimeModel;
    private IWorker worker;

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
            return;
        }

        ProcessStaticImage(testImage);
    }

    void ProcessStaticImage(Texture2D image)
    {
        Tensor inputTensor = Preprocess(image);
        worker.Execute(inputTensor);
        Tensor outputTensor = worker.PeekOutput();

        ProcessOutput(outputTensor);

        inputTensor.Dispose();
        outputTensor.Dispose();
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

        int detections = outputTensor.shape[2] / 21;

        for (int i = 0; i < detections; i++)
        {
            float confidence = outputTensor[0, 0, i * 21, 0];

            if (confidence <= 0.2)
                continue;

            int classId = -1;
            float maxProb = 0f;

            for (int j = 1; j < 21; j++)
            {
                float prob = outputTensor[0, 0, i * 21 + j, 0];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    classId = j - 1;
                }
            }

            if (classId != -1)
            {
                detected = true;
                string detectedClassName = classLabels[classId];
                // Only log if an object is detected to avoid flooding the console
                Debug.Log($"Object Detected: {detectedClassName} | Confidence: {confidence} | Probability: {maxProb}");
                DisplayClassName(detectedClassName);
                break; // Display the first detected class for simplicity
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
        labelText.text = className;
    }
}
