Release steps
===

Suppose you are publishing a new version 0.x.

Linux
---

First, build the wheels.

```bash
bash build_release.sh 0.x
```

The script assumes there is a 0.x branch or tag in the git repository. If successful, wheels are stored in the `cpu-release` and `gpu-release` folders.

Then, upload the wheels to S3 or pypi.
