from collections import defaultdict
from random import randint
from urllib.parse import urlparse
import pandas as pd

class Preresolver:
  def __init__(self, preprocessed_domain_file) -> None:
    df = pd.read_csv(preprocessed_domain_file, sep=" ", header=None, names=["domain", "ip"])

    # read massdns fine
    self.domain_to_ips = defaultdict(list)
    for domain, ip in zip(df["domain"], df["ip"]):
      self.domain_to_ips[domain].append(ip)

  def url_to_ip(self, url):
    domain = urlparse(url).netloc
    return self.domain_to_ip(domain)

  def domain_to_ip(self, domain):
    ips = self.domain_to_ips[domain]
    if len(ips) == 0:
      return None
    return ips[randint(0, len(ips)-1)]

  def __call__(self, df):
    df["ip"] = df["url"].apply(lambda url: self.url_to_ip(url))
    return df
