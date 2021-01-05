# machine_translation
Machine Translation System using different architectures and implemented in pytorch

# Data
Used the [European Parliament Proceedings Parallel Corpus](http://www.statmt.org/europarl/)

# Build Training Image
from outside the `machine_translation` directory
`docker build -f machine_translation/Dockerfile -t machine-translation:0.1 .`
