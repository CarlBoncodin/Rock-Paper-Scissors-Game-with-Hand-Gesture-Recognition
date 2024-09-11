# Rock Paper Scissors Game with Hand Gesture Recognition

This project developed a Rock, Paper, Scissors game using hand gesture recognition to create an interactive experience where players use their hands to play against a computer. The system uses OpenCV and TensorFlow to capture and process hand gestures through a webcam, with the computer opponent making moves based on a pre-trained neural network.

Objectives
-

- Implement a real-time hand gesture recognition system for Rock, Paper, Scissors.
- Train a neural network to classify gestures accurately.
- Integrate the gesture recognition system with a functional game interface.

Methodology
-
Data Collection and Preparation:
-bA dataset of hand gestures for Rock, Paper, and Scissors was collected. The dataset was augmented with techniques such as flipping, rotation, and zooming to enhance the model’s robustness and generalization.

Model Training:
- The pre-trained Xception model, known for its efficiency in image classification tasks, was employed for gesture recognition. This model was fine-tuned using the augmented dataset with transfer learning. The training process involved using the Adam optimizer and categorical cross-entropy loss function to adapt the model to classify gestures into Rock, Paper, or Scissors.

Real-time Gesture Recognition:
- OpenCV was used to capture live video feed from the webcam. MediaPipe's hand tracking module detected hand landmarks in each frame, which was then processed to extract the region of interest (ROI) around the hand. This ROI was passed to the neural network to classify the hand gesture.

Game Logic Integration:
- The classified gestures from both the player and the computer (which randomly selects gestures) were compared to determine the winner of each round. The game logic was implemented to reflect the rules of Rock, Paper, Scissors and provide feedback to the player based on the outcome.

Results
-

Gesture Recognition Accuracy:
- The system successfully identified hand gestures for Rock, Paper, and Scissors with high accuracy. For gestures like Rock and Paper, the model performed well, consistently detecting the correct hand symbols as shown in the examples where the computer's move and the player's gesture matched accurately.

Challenges with Scissors Gesture:
- The recognition of the Scissors gesture was more challenging due to the complexity of the hand shape, which involves two protruding fingers. The model occasionally struggled with background noise and variations in hand positioning, leading to less consistent recognition compared to Rock and Paper.

Training Performance:
- The model achieved a training accuracy of [insert accuracy]% and a validation accuracy of [insert accuracy]% on the dataset. This indicates that the model was well-trained, though some areas, such as gesture complexity and background variation, showed room for improvement.

Game Functionality:
- The integration of gesture recognition with game logic was successful, allowing players to engage in real-time matches against the computer. The feedback system accurately reflected the game outcomes, demonstrating that the system can handle live interaction effectively.


