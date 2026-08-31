local utils = import 'utils.libsonnet';

{
  uses_user_defaults: true,
  local top = self,
  project_name: 'smlab',
  primary_module: 'smlab',
  description: 'Generate StepMania dance-single charts from audio using machine learning.',
  keywords: ['audio', 'command line', 'dance dance revolution', 'machine learning', 'simfile', 'stepmania'],
  version: '0.0.1',
  want_main: true,
  want_flatpak: true,
  // Librosa requires 3.12.
  supported_python_versions: ['3.12', '3.13', '3.14'],
  // torch publishes no wheels for Intel macOS or Windows on ARM, and no source distribution to
  // fall back on, so those two can only ever fail to build.
  supported_platforms: ['linux', 'macos-arm64', 'windows-x86_64'],
  publishing+: { flathub: 'sh.tat.smlab' },
  // torch's Linux wheels carry the CUDA runtime, so a bundle that takes what PyPI serves builds
  // out to 2.7 GB, over GitHub's 2 GiB limit for a release asset. Every bundled format is
  // therefore held to the CPU wheels, which is a real limitation of them: a GPU is available to
  // an installation from PyPI, and not to these.
  local torch_cpu_index = 'https://download.pytorch.org/whl/cpu',
  appimage+: {
    requirements_options: ['--extra-index-url %s' % torch_cpu_index],
  },
  snapcraft+: {
    parts+: {
      [top.project_name]+: {
        'build-environment': [{ PIP_EXTRA_INDEX_URL: torch_cpu_index }],
      },
    },
  },
  flatpak+: {
    modules: [
      super.modules[0] {
        'build-commands': [
          'pip3 install --prefix=/app uv',
          '/app/bin/uv pip install --torch-backend=cpu --prefix=/app .',
        ],
      },
    ],
  },
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
  prettierignore+: ['*.pt', '*.sha256'],
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
        'librosa>=1.0.0',
        'mutagen>=1.48.1',
        'numpy>=2.5.2',
        'pillow>=12.3.0',
        'platformdirs>=4.11.3',
        'scipy>=1.18.0',
        'soundfile>=0.14.0',
        'torch>=2.11.0',
      ],
      'optional-dependencies'+: {
        // Demucs pins numpy below 2 on Intel macOS.
        stems: ["demucs>=4.1.0; sys_platform != 'darwin' or platform_machine != 'x86_64'"],
      },
    },
    tool+: {
      coverage+: {
        local omit = ['%s/typing.py' % top.primary_module],
        report+: { omit+: omit },
        run+: { omit+: omit },
      },
      ruff+: {
        lint+: {
          isort+: {
            'known-first-party': [top.primary_module],
            'section-order': [
              'future',
              'standard-library',
              'third-party',
              'first-party',
              'local-folder',
            ],
          },
        },
      },
      uv+: {
        'cache-dir': uv_cache_dir,
        // Intel macOS is left out: demucs pins numpy below 2 there.
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
