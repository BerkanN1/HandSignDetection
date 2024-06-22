import UIKit
import AVFoundation
import Vision
import CoreML
import SwiftUI

class ViewController: UIViewController {
    private let fingerDetectModel: HandSign = {
        do {
            let config = MLModelConfiguration()
            return try HandSign(configuration: config)
        } catch {
            fatalError("Couldn't create model")
        }
    }()
    
    private let handPoseRequest: VNDetectHumanHandPoseRequest = {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1
        
        return request
    }()
    
    private let videoDataOutputQueue = DispatchQueue(
        label: "CameraFeedOutput",
        qos: .userInteractive
    )
    
    private var csvString = ""
    private var lineCount = 0
    private var isAnimationWaiting = false
    private var iconImageView = UIImageView()
    private let cameraView = UIView(frame: .zero)
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    
    private var predictions: [String] = []
    
    private lazy var predictionLabel: UILabel = {
        let label = UILabel()
        label.textColor = .black
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 20)
        return label
    }()
    
    private let clearButton: UIButton = {
        let button = UIButton()
        button.setTitle("Sil", for: .normal)
        button.setTitleColor(.black, for: .normal)
        button.backgroundColor = .lightGray
        button.layer.cornerRadius = 8
        button.addTarget(self, action: #selector(clearPredictions), for: .touchUpInside)
        return button
    }()
    
    private let detectionTitle: UILabel = {
        let label = UILabel()
        label.text = "Detection"
        label.textColor = .black
        label.font = UIFont.boldSystemFont(ofSize: 30)
        label.textAlignment = .left
        return label
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setup()
        addCameraInput()
        showCameraFeed()
        captureSession.startRunning()
        getCameraFrames()
        
        // Detection Title'u ekleyelim
        view.addSubview(detectionTitle)
        detectionTitle.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            detectionTitle.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            detectionTitle.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20)
        ])
        
        // Prediction Label ve Clear Button'u dikey StackView içinde ekleyelim
        let stackView = UIStackView(arrangedSubviews: [predictionLabel, clearButton])
        stackView.axis = .vertical
        stackView.alignment = .center
        stackView.spacing = 10
        
        view.addSubview(stackView)
        
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: cameraView.bottomAnchor, constant: 20),
            stackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            clearButton.widthAnchor.constraint(equalToConstant: 100),
            clearButton.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Arka plan için gradient ekleyelim
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [UIColor.gray.cgColor, UIColor.red.cgColor]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.frame = view.bounds
        view.layer.insertSublayer(gradientLayer, at: 0)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = cameraView.bounds
        view.layer.sublayers?.first?.frame = view.bounds // GradientLayer'ın boyutunu güncelle
    }
    
    private func setup() {
        csvString = csvString.appending("wristx,wristy,thumbCMCx,thumbCMCy,thumbMPx,thumbMPy,thumbIPx,thumbIPy,thumbTipx,thumbTipy,indexMCPx,indexMCPy,indexPIPx,indexPIPy,indexDIPx,indexDIPy,indexTipx,indexTipy,middleMCPx,middleMCPy,middlePIPx,middlePIPy,middleDIPx,middleDIPy,middleTipx,middleTipy,ringMCPx,ringMCPy,ringPIPx,ringPIPy,ringDIPx,ringDIPy,ringTipx,ringTipy,littleMCPx,littleMCPy,littlePIPx,littlePIPy,littleDIPx,littleDIPy,littleTipx,littleTipy,result\n")
       
        view.backgroundColor = .white
        view.addSubview(cameraView)
        cameraView.clipsToBounds = true
        cameraView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            cameraView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.87), // Ekran genişliğinin %80'i
            cameraView.heightAnchor.constraint(equalTo: cameraView.widthAnchor, multiplier: 1.75), // Yükseklik genişliğin 1.5 katı
            cameraView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraView.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
    
    private func addCameraInput() {
        guard let device = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera],
                mediaType: .video,
                position: .front).devices.first else {
            fatalError("No back camera device found, please make sure to run SimpleLaneDetection in an iOS device and not a simulator")
        }
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        self.captureSession.addInput(cameraInput)
    }
    
    private func showCameraFeed() {
        self.previewLayer.videoGravity = .resizeAspectFill
        self.cameraView.layer.addSublayer(self.previewLayer)
    }
    
    private func getCameraFrames() {
        let dataOutput = AVCaptureVideoDataOutput()
        
        if captureSession.canAddOutput(dataOutput) {
            captureSession.addOutput(dataOutput)
            // Add a video data output.
            dataOutput.alwaysDiscardsLateVideoFrames = true
            dataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            dataOutput.alwaysDiscardsLateVideoFrames = true
            dataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            
            dataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            dataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        }
        
        captureSession.commitConfiguration()
    }
    
    func startAnimationIfNeeded(bottomPoint: CGPoint, result: Int) {
        if isAnimationWaiting {
            return
        }
        
        isAnimationWaiting = true
        iconImageView.removeFromSuperview()
        
        var predictionText = ""
        switch result {
        case 1:
            predictionText = "A"
        case 2:
            predictionText = "B"
        case 3:
            predictionText = "C"
        case 4:
            predictionText = "D"
        case 5:
            predictionText = "E"
        case 6:
            predictionText = "F"
        case 7:
            predictionText = "G"
        case 8:
            predictionText = "H"
        case 9:
            predictionText = "I"
        case 10:
            predictionText = "K"
        case 11:
            predictionText = "L"
        case 12:
            predictionText = "M"
        case 13:
            predictionText = "N"
        case 14:
            predictionText = "O"
        case 15:
            predictionText = "P"
        case 16:
            predictionText = "R"
        case 17:
            predictionText = "S"
        case 18:
            predictionText = "T"
        case 19:
            predictionText = "U"
        case 20:
            predictionText = "V"
        case 21:
            predictionText = "Y"
        default:
            predictionText = ""
        }
        
        predictions.append(predictionText)
        
        iconImageView.frame = CGRect(x: bottomPoint.x, y: bottomPoint.y, width: 30, height: 30)
        iconImageView.translatesAutoresizingMaskIntoConstraints = true
        cameraView.addSubview(iconImageView)
        
        UIView.animate(withDuration: 0.5) { [weak self] in
            self?.iconImageView.frame.size = CGSize(width: 150, height: 150)
        } completion: { _ in
            UIView.animate(withDuration: 0.5) {
                self.iconImageView.transform = CGAffineTransform(scaleX: 0.1, y: 0.1)
                self.iconImageView.center = CGPoint(x: bottomPoint.x, y: -30)
            } completion: { _ in
                self.isAnimationWaiting = false
                // Tahmin sonuçlarını güncelleyelim
                self.updatePredictionLabel()
            }
        }
    }
    
    private func updatePredictionLabel() {
        // Tahmin sonuçlarını birleştirerek etikete gösterelim
        DispatchQueue.main.async {
            self.predictionLabel.text = "Tahmin Sonuçları: \(self.predictions.joined(separator: ""))"
        }
    }
    
    @objc private func clearPredictions() {
        // Tahmin sonuçlarını temizleyelim
        predictions.removeAll()
        updatePredictionLabel()
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
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
    func createCSVFile(fingers: [VNHumanHandPoseObservation.JointName : VNRecognizedPoint]) {
        let landmarks: [VNHumanHandPoseObservation.JointName] = [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip, .indexMCP, .indexPIP, .indexDIP, .indexTip, .middleMCP, .middlePIP, .middleDIP, .middleTip, .ringMCP, .ringPIP, .ringDIP, .ringTip, .littleMCP, .littlePIP, .littleDIP, .littleTip]
        var line = ""
        landmarks.forEach { type in
            let strx = fingers[type]?.x ?? 0
            line = line.appending(String(strx))
            line = line.appending(",")
            let stry = fingers[type]?.y ?? 0
            line = line.appending(String(stry))
            line = line.appending(",")
        }
        
        lineCount += 1
        
        line = line.appending("1\n") //change 1 when every hand sign changed

        csvString = csvString.appending(line)

        if lineCount == 400 {
            let fileManager = FileManager.default
            do {
                let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false)
                let fileURL = path.appendingPathComponent("FingerPointsY.csv")
                try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
                print("ok")
            } catch let error {
                print(error)
            }
        }
    }

    
    func detectFingerIfNeeded(fingers: [VNHumanHandPoseObservation.JointName : VNRecognizedPoint]) {
        guard let modelOutput = try? fingerDetectModel.prediction(
            wristx: fingers[.wrist]?.x ?? 0,
            wristy: fingers[.wrist]?.y ?? 0,
            thumbCMCx: fingers[.thumbCMC]?.x ?? 0,
            thumbCMCy: fingers[.thumbCMC]?.y ?? 0,
            thumbMPx: fingers[.thumbMP]?.x ?? 0,
            thumbMPy: fingers[.thumbMP]?.y ?? 0,
            thumbIPx: fingers[.thumbIP]?.x ?? 0,
            thumbIPy: fingers[.thumbIP]?.y ?? 0,
            thumbTipx: fingers[.thumbTip]?.x ?? 0,
            thumbTipy: fingers[.thumbTip]?.y ?? 0,
            indexMCPx: fingers[.indexMCP]?.x ?? 0,
            indexMCPy: fingers[.indexMCP]?.y ?? 0,
            indexPIPx: fingers[.indexPIP]?.x ?? 0,
            indexPIPy: fingers[.indexPIP]?.y ?? 0,
            indexDIPx: fingers[.indexDIP]?.x ?? 0,
            indexDIPy: fingers[.indexDIP]?.y ?? 0,
            indexTipx: fingers[.indexTip]?.x ?? 0,
            indexTipy: fingers[.indexTip]?.y ?? 0,
            middleMCPx: fingers[.middleMCP]?.x ?? 0,
            middleMCPy: fingers[.middleMCP]?.y ?? 0,
            middlePIPx: fingers[.middlePIP]?.x ?? 0,
            middlePIPy: fingers[.middlePIP]?.y ?? 0,
            middleDIPx: fingers[.middleDIP]?.x ?? 0,
            middleDIPy: fingers[.middleDIP]?.y ?? 0,
            middleTipx: fingers[.middleTip]?.x ?? 0,
            middleTipy: fingers[.middleTip]?.y ?? 0,
            ringMCPx: fingers[.ringMCP]?.x ?? 0,
            ringMCPy: fingers[.ringMCP]?.y ?? 0,
            ringPIPx: fingers[.ringPIP]?.x ?? 0,
            ringPIPy: fingers[.ringPIP]?.y ?? 0,
            ringDIPx: fingers[.ringDIP]?.x ?? 0,
            ringDIPy: fingers[.ringDIP]?.y ?? 0,
            ringTipx: fingers[.ringTip]?.x ?? 0,
            ringTipy: fingers[.ringTip]?.y ?? 0,
            littleMCPx: fingers[.littleMCP]?.x ?? 0,
            littleMCPy: fingers[.littleMCP]?.y ?? 0,
            littlePIPx: fingers[.littlePIP]?.x ?? 0,
            littlePIPy: fingers[.littlePIP]?.y ?? 0,
            littleDIPx: fingers[.littleDIP]?.x ?? 0,
            littleDIPy: fingers[.littleDIP]?.y ?? 0,
            littleTipx: fingers[.littleTip]?.x ?? 0,
            littleTipy: fingers[.littleTip]?.y ?? 0
        ) else { return }
        
        DispatchQueue.main.async {
            let wristPoint = fingers[.wrist]
            let bottomPoint = self.previewLayer.layerPointConverted(
                fromCaptureDevicePoint: CGPoint(
                    x: wristPoint?.x ?? 0,
                    y: 1 - (wristPoint?.y ?? 0)
                )
            )
            self.startAnimationIfNeeded(
                bottomPoint: bottomPoint,
                result: Int(modelOutput.result)
            )
        }
    }
}
