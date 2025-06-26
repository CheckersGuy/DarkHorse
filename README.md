# ![logo](https://github.com/user-attachments/assets/46571f61-95c9-4a27-942d-f4e438d76a19) 

# DarkHorse – Checkers Engine

**DarkHorse** is a checkers engine powered by three neural networks to enhance its decision-making:

- **Evaluation Network** – Assesses board positions to determine their strategic value.  
- **Game-Length Estimator** – Predicts how long a game might last from a given position. This helps the engine prefer positions where it is still making progress.  
- **Policy Network** – Produces a probability distribution over possible moves, improving move-ordering during search.

The neural networks are embedded directly in the binary, so there's no need to download any additional files. Occasionally, small updates may be released if improved networks become available.

This project began quite some time ago, but I eventually stopped working on it seriously. It became more of a playground for experimenting with ideas on how to design better neural networks for checkers or other board games. It turned out to be a great learning experience—particularly in understanding how neural networks can be applied to game-tree search. Most of the development time was likely spent creating new datasets, fixing bugs in the game-generation code, and then generating yet another dataset.

#### Thanks to all the other engine authors whose work served as a valuable inspiration throughout this project.

- **Martin Fierz** — Author of the powerful checkers program [*Cake*](http://www.fierz.ch/cake.php) and the excellent [CheckerBoard GUI](http://www.fierz.ch/checkerboard.php). The [*story*](http://www.fierz.ch/cake186.php) behind the making of Cake 1.86-1.89 was what finally motivated to finish the project and create this release.
- **Ed Gilbert** — Creator of the strong checkers engine  [*Kingsrow*](https://edgilbert.org/Checkers/KingsRow.htm), who also created checkers endgame tablebases.  
- **Jonathan Kreuzer** — Developer of the advanced checkers engine  [*GuiNN*](https://github.com/jonkr2/GuiNN_Checkers), which also uses a neural network for position evaluation.



## You can find the engine on the release page !

---

*Feedback and contributions are welcome !*
