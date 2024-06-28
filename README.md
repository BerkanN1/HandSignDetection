
# Hand Sign Detection

This project aims to convert American sign language into latin alphabet using coreml.  


## Appendix

This project was developed using createml tabluar classification. If you want to create your own dataset you can use the create dataset step. If you don't want to create your own dataset, you can clone the project and use it. 


## Create Dataset
İf you need to create your own dataset you can include the comment line in the code.

```swift
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        
        do {
            try handler.perform([handPoseRequest])
            guard let results = handPoseRequest.results?.first else { return }
            let fingers = try results.recognizedPoints(.all)
            detectFingerIfNeeded(fingers: fingers)
            //createCSVFile(fingers: fingers)
        } catch {
            print(error.localizedDescription)
        }
    }
```


## Train Model

- Open Create Ml

- Choose tabluar classification 
- Add your dataset and Train

## Demo



https://github.com/BerkanN1/HandSignDetection/assets/103366156/0d681fdf-0747-4951-9d23-f12b90e0dcf3



