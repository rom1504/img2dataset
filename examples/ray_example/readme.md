# Parallelizing Img2Dataset using Ray 
If you do not want to set up a PySpark cluster, you can also set up a ray cluster, functionally they are 
close to the same but ray handles a larger amount of tasks better and doesn't have the "staged" nature of 
Spark which is great if you have a large queue of tasks and don't want to be vulnerable to the stragglers in each batch.
The tooling to set up a Ray cluster on AWS is slightly better at the time of writing this document (Jan 2023)

## Instructions for running a large img2dataset job on a ray cluster on AWS
First install ray:
``` pip install ray ```

If you are on AWS you can spin up a ray cluster this way:

``` ray up cluster_minimal.yaml ```

Then you can run your job:
```ray submit cluster_minmal.yaml ray_example.py -- --url_list <url_list> --out_folder <out_folder>```

Using the above code I was able to achieve a maximum download rate of 220,000 images/second on a cluster of 100 m5.24xlarge (9600 cores).

