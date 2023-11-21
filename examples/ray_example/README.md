## Instructions for running a large img2dataset job on a ray cluster on AWS
First install ray:
``` pip install ray ```

If you are on AWS you can spin up a ray cluster this way:

``` ray up cluster_minimal.yaml ```

Then you can run your job:
```ray submit cluster_minimal.yaml ray_example.py -- --url_list <url_list> --out_folder <out_folder>```

You may also setup a ray cluster by following https://docs.ray.io/en/latest/cluster/getting-started.html

