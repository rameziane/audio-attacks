# Attacking Whisper : whitebox adversarial attacks on speech-to-text models


## Motivation
Many real world applications nowadays are integrating Automatic Speech Recognition (ASR) as part of their workflow. 
Thus it becomes important to look at how these models can be attacked.

Here are some examples :

* Financial Theft : An attacker could embed an inaudible command like "Alexa, buy a £500 gift card and send it to this email" into the background music of a YouTube video or podcast you're listening to. If your smart speaker is within earshot, it would execute the purchase without you ever hearing the command.

* Physical Security Breach : A command like "Hey Google, unlock the front door" or "Siri, disable the security cameras" could be hidden in a TV commercial. This could be timed to facilitate a burglary, with the attacker knowing exactly when your smart lock will disengage.

* Covert Surveillance : An attacker could broadcast a hidden command like "Siri, call [attacker's number] and turn on speakerphone." This would effectively turn your phone into a hot mic, allowing the attacker to eavesdrop on your private conversations, business meetings, or family moments.

The core danger of this type of attack lies in its stealth. Because the commands are imperceptible to humans, victims have no idea they are being manipulated, making it an incredibly powerful tool for fraud, espionage and large-scale disruption.

This is why defending against adversarial attacks is a major focus in modern AI security research. While the attacks implemented in this project would not work for over-the-air attacks, it is nevertheless dangerous.


## Goal
The goal of this project is to attack Whisper, one of the most popular ASR models. We distinguish two types of attacks : 

* Untargeted attacks : make the model mistranscribe predictions.

* Targeted attacks : transcribe a specific sentence of the attacker's choice.

## Source

All attacks in this repo are based on the paper "There is more than one kind of robustness: Fooling Whisper with adversarial
examples" by Raphael Olivier and Bhiksha Raj.

