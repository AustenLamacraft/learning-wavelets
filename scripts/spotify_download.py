#!/usr/bin/env python
import argparse
import base64
from datetime import datetime, timedelta
import json
import math
import os
import re
import shutil
from urllib.request import urlretrieve
from urllib.parse import urlparse

import requests
import tqdm


def iter_batches(iterable, batch_size):
    """
    Generated batches of `batch_size` from `iterable`.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class SpotifyApiClient:
    """
    Client for the public Spotify web API (https://developer.spotify.com/documentation/web-api/).

    Notes
    -----
    You need to register a developer account on the Spotify web API and register an application
    here: https://developer.spotify.com/dashboard/applications You will be given a `client_id`
    and `client_secret` that can be used to authenticate the client.
    """
    def __init__(self, client_id, client_secret, base_url='https://api.spotify.com/v1/'):
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = None
        self.expires_at = None
        self.base_url = base_url

    @property
    def access_token(self):
        """
        Get an valid access token. The token is renewed automatically if necessary.
        """
        if self.expires_at is None or self.expires_at < datetime.now() - timedelta(seconds=60):
            encoded = base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()
            response = requests.post(
                'https://accounts.spotify.com/api/token',
                data={'grant_type': 'client_credentials'},
                headers={'Authorization': f'Basic {encoded}'})
            response.raise_for_status()
            response = response.json()
            self._access_token = response['access_token']
            self.expires_at = datetime.now() + timedelta(seconds=response['expires_in'])

        return self._access_token

    def get(self, url, params, **kwargs):
        """
        Get a response from the Spotify web API.
        """
        parsed = urlparse(url)
        if not parsed.scheme:
            url = os.path.join(self.base_url, url)
        headers = kwargs.pop('headers', {})
        headers.setdefault('Authorization', f'Bearer {self.access_token}')
        response = requests.get(url, params, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()


URI_PATTERN = re.compile(r'spotify:track:\w{22}$')


def __main__(args=None):
    """
    Entrypoint for command line interface.
    """
    parser = argparse.ArgumentParser('spotify_download', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output-directory', '-o', help='directory for audio files', default=os.getcwd())
    parser.add_argument('--file', '-f', help='file containing track uris')
    parser.add_argument('--client-id', '-c', help='Spotify web API client id (defaults to environment variable SPOTIFY_CLIENT_ID)',
                        default=os.environ.get('SPOTIFY_CLIENT_ID'))
    parser.add_argument('--client-secret', '-s', help='Spotify web API client secret (defaults to environment variable SPOTIFY_CLIENT_SECRET)',
                        default=os.environ.get('SPOTIFY_CLIENT_SECRET'))
    parser.add_argument('uris', help='one or more track uris', nargs='*')
    args = parser.parse_args(args)

    assert args.client_id, "client id is missing; it must be specified as a command line argument "\
        "or the environment variable SPOTIFY_CLIENT_ID"
    assert args.client_secret, "client id is missing; it must be specified as a command line " \
        "argument or the environment variable SPOTIFY_CLIENT_SECRET"

    os.makedirs(args.output_directory, exist_ok=True)

    # Get all uris
    uris = []
    if args.uris:
        uris.extend(args.uris)

    if args.file:
        with open(args.file) as fp:
            uris.extend(fp.readlines())

    uris = set(uri.strip() for uri in uris)

    # Validate the uris
    for uri in uris:
        if not URI_PATTERN.match(uri):
            raise ValueError("%s is not a valid track uri" % uri)

    # Get all the file ids
    downloaded = set()
    client = SpotifyApiClient(args.client_id, args.client_secret)
    with tqdm.tqdm(total=len(uris)) as progress:
        for batch in iter_batches(uris, 50):
            ids = [uri.split(':')[2] for uri in batch]
            response = client.get('tracks', {'ids': ','.join(ids)})

            for track in response['tracks']:
                # Check if the url is available
                if not track:
                    continue
                preview_url = track.get('preview_url')
                if not preview_url:
                    continue

                # Split of the identifier
                preview_url, _ = preview_url.split('?')

                path = os.path.join(args.output_directory, 'spotify_track_%s.mp3' % track['id'])
                # Skip existing files
                if not os.path.isfile(path):
                    try:
                        tempfile, _ = urlretrieve(preview_url)
                        shutil.move(tempfile, path)
                    except:
                        if os.path.isfile(path):
                            os.unlink(path)
                        raise
                downloaded.add('spotify:track:%s' % track['id'])
                progress.update()

    print("Downloaded %d of %d previews." % (len(downloaded), len(uris)))
    missing = uris - downloaded
    if missing:
        print("Failed to download %d previews: %s" % (len(missing), ", ".join(missing)))


if __name__ == "__main__":
    __main__()
