local utils = import 'utils.libsonnet';

{
  uses_user_defaults: true,
  local top = self,
  project_name: 'smlab',
  primary_module: 'smlab',
  description: 'Generate StepMania dance-single charts from audio using machine learning.',
  keywords: ['audio', 'command line', 'dance dance revolution', 'machine learning', 'simfile', 'stepmania'],
  version: '0.0.0',
  want_main: true,
  want_flatpak: true,
  // Librosa requires 3.12.
  supported_python_versions: ['3.12', '3.13', '3.14'],
  publishing+: { flathub: 'sh.tat.smlab' },
  local uv_cache_dir = '.uv-cache',
  shared_ignore+: [
    '*.npy',
    '*.npz',
    '/%s/' % uv_cache_dir,
    '/cache/',
    '/checkpoints/',
    '/out/',
  ],
  // Checkpoints are binary but are shipped, so they stay out of shared_ignore.
  prettierignore+: ['*.pt'],
  pyproject+: {
    project+: {
      classifiers: utils.pyprojectClassifiers(top, [
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Games/Entertainment',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ]),
      dependencies+: [
        'huggingface-hub>=1.27.0',
        'librosa>=1.0.0',
        'mutagen>=1.48.1',
        'numpy>=2.5.2',
        'pillow>=12.3.0',
        'scipy>=1.18.0',
        'soundfile>=0.14.0',
        'torch>=2.11.0',
      ],
      'optional-dependencies'+: {
        // Demucs pins numpy below 2 on Intel macOS, which this project cannot
        // satisfy, so separation is unavailable there.
        stems: ["demucs>=4.1.0; sys_platform != 'darwin' or platform_machine != 'x86_64'"],
      },
    },
    tool+: {
      coverage+: {
        report+: { omit+: ['%s/typing.py' % top.primary_module] },
        run+: { omit+: ['%s/typing.py' % top.primary_module] },
      },
      uv+: {
        'cache-dir': uv_cache_dir,
        // Intel macOS is left out: demucs pins numpy below 2 there, which this
        // project cannot satisfy.
        environments: [
          "sys_platform == 'linux'",
          "sys_platform == 'darwin' and platform_machine == 'arm64'",
          "sys_platform == 'win32'",
        ],
      },
    },
  },
  docs_conf+: {
    config+: {
      intersphinx_mapping+: {
        bascom: ['https://bascom.readthedocs.io/en/latest/', null],
        click: ['https://click.palletsprojects.com/en/stable/', null],
        librosa: ['https://librosa.org/doc/latest/', null],
        numpy: ['https://numpy.org/doc/stable/', null],
        scipy: ['https://docs.scipy.org/doc/scipy/', null],
        torch: ['https://pytorch.org/docs/stable/', null],
      },
    },
  },
}
