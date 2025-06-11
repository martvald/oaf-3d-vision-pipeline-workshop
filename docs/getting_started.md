# Getting Started

To host the website yourself you have to follow the instuctions below. As we go along with the workshops you will be coding the missing parts of the code and publishing your changes to the website.

## Creating your own website

### 1. Fork the Repository

This step will create a copy of the repository in your GitHub account:

1. Navigate to the [repository](https://github.com/martvald/oaf-3d-vision-pipeline-workshop).
2. Click the **`Fork`** button in the top-right corner of the page.
3. Enter a name for your forked repository (e.g., `pixel-to-point-cloud`) and optionally add a description.

   ![Fork repository](../test_data/forking/fork_repository.png)

4. You’ll be redirected to your fork. If not, visit your GitHub profile and click on the newly forked repository.

### 2. Configure GitHub Pages

We need to set up GitHub Pages for your forked repository to host the website:

1. In your forked repository, go to the **`Settings`** tab.
2. Scroll to the **`Pages`** section.
3. Select **GitHub Actions** as the source.
4. (Optional) Set up a custom domain. By default, your site will be available at:

    ```bash
    https://<your-username>.github.io/<your-repo-name>/
    ```

    ![Configure GitHub Pages](../test_data/forking/configure_github_pages.png)

### 3. Enable GitHub Actions Workflows

To publish the website, we need to use the GitHub Actions workflow in the repository. These are disabled by default in forked repositories:

1. Navigate to the **`Actions`** tab.
2. Since workflows are disabled by default in forked repositories, enable them by clicking: **"I understand my workflows, go ahead and enable them."**

   ![Enable workflows](../test_data/forking/enable_github_actions.png)

### 4. Make Changes and Create a Pull Request (PR)

Now everything is set up for you to make changes to the repository and publish them. The forked repo has some paths that should be updated so the links work correctly. To do this, follow the steps below:

#### a. Set Up the Repository Locally

````{margin}
```{warning}
I recommend to make a seperate folders for cloning repositories to, make sure
this folders is not within a synced folder like `OneDrive` or `iCloud`. These
are the locations I use personally:
- Unix: `/Users/{user_name}/Code`
- Windows: `C:/Code`
```
````

1. Clone your forked repository to your local machine:

    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    ```

2. Create a new branch for your changes:

    ```bash
    git checkout -b workshop-delivery
    ```

#### b. Update Repository Links

1. Update the following files:

   - `README.md`:

      Replace the link to the original site:

      ```markdown
      https://martvald.github.io/oaf-3d-vision-pipeline-workshop/
      ```

      with your GitHub Pages link:

      ```markdown
      https://<your-username>.github.io/<your-repo-name>/
      ```

   - `landing.py`:

      Replace all instances of:

      ``` bash
      martvald/oaf-3d-vision-pipeline-workshop
      ```

      with your user and repository name:

      ``` bash
      <your-username>/<your-repo-name>
      ```

   - `docs/getting_started.md`:

      This is the file you are currently reading. You can update this as well if you to show how to fork from you repository. To do this replace all instances of:

      ``` bash
      martvald
      ```

      with your user name:

      ``` bash
      <your-username>
      ```

      and all instances of:

      ``` bash
      oaf-3d-vision-pipeline-workshop
      ```

      with your repository name:

      ``` bash
      <your-repo-name>
      ```

2. Commit and push your changes:

   ```bash
   git commit -am "Update links to my own repository"
   git push origin workshop-delivery
   ```

3. Optionally, you can make additional changes to the website content to include your own project changes. You can also add changes in later PRs and the website will update automatically.

4. Open a Pull Request (PR) from your `workshop-delivery` branch to the `main` branch of your forked repository.
   - Ensure the target is your fork, not the original repository.

### 5. Publish Your Changes

1. Wait for the GitHub Actions workflow to complete on your PR.
   - **No branch protection rules are applied by default, so there is no protection against merging the PR before the workflows finish.**
2. Review the generated static site:
   - Go to the **`Actions`** tab and select the documentation job.
   - Under **`Summary`**, download the artifact containing the static HTML files.
   - Open the `index.html` file locally to preview the site.
3. If everything looks good, merge the PR into the `main` branch.
4. The publish job will automatically deploy your changes to GitHub Pages.
5. Once the job finishes, visit your live site at:

   ``` bash
   https://<your-username>.github.io/<your-repo-name>/
   ```

When deployed successfully, you should be able to see the status in **`Pages`** section of your repository **`Settings`**:

![Deployed site](../test_data/forking/publised_pages.png)

## Using Pants

Pants 2 is the build system configured for this repository. You can find the [prerequisites here](https://www.pantsbuild.org/2.25/docs/getting-started/prerequisites) and their [installation instructions here](https://www.pantsbuild.org/2.25/docs/getting-started/installing-pants).

### Common commands

| Purpose                         | Command                             |
|---------------------------------|-------------------------------------|
| List documented targets         | `pants list --documented ::`        |
| Lint                            | `pants lint ::`                     |
| Auto‑fix lint issues            | `pants fix ::`                      |
| Static type check               | `pants check ::`                    |
| Run tests                       | `pants test ::`                     |
| Build docs / website            | `pants run docs:build_docs`         |
| Build dists (wheel/sdist)       | `pants package ::`                  |

### Set up your Python environment

I recommend running code from a virtual environment. This assures that changes,
installations and more does not affect the whole pc, only the current environment.

`````{tab-set}
````{tab-item} Using Pants
Use pants to create a virtual environment with all the dependencies and editable packages.
```bash
pants export --resolve=default
```
Activate the environment.
```bash
source dist/export/python/virtualenvs/default/3.11.10/bin/activate      # Unix/macOS
dist\export\python\virtualenvs\default\3.11.10\Scripts\activate         # Windows
```
````

````{tab-item} Manual Python setup
You can create a virtual environment manually by running the following command.
```bash
python -m venv ENV
```
Activate the environment.
```bash
source ENV/bin/activate      # Unix/macOS
ENV/Scripts/activate         # Windows
```
Install dependencies:
```bash
pip install -r third_party/python/requirements.txt
```
Append to python path to get the workshop packages
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Unix/macOS
set PYTHONPATH=%PYTHONPATH%;$(pwd)     # Windows
```
````
`````

### 4. Open editor at repo root in the virtual environment:

For these workshops, I will use the free Visual Studio Code (VS Code), a popular
code editor from Microsoft that offers a wide range of plugins and features. I
find VS Code to be an excellent choice for Python programming, but you are free
to use your preferred editor if you already have a favorite. It will be up to you
to ensure that you know how to set up and use your chosen editor effectively.

To open the repository in VS Code:

`````{tab-set}
````{tab-item} Windows
Open a terminal at the repo root and enter your vitual environment:
```shell
dist\export\python\virtualenvs\default\3.11.10\Scripts\activate
```

Open VS Code at repo root:
```shell
code .
```
````

````{tab-item} Unix
Open a terminal at the repo root and enter your vitual environment:
```shell
source dist/export/python/virtualenvs/default/3.11.10/bin/activate
```

Open VS Code at repo root:
```shell
code .
```
````
`````

## Repository Structure

- `workshops/`: Contains the jupyter notebooks for each workshop
- `oaf_vision_3d/`: Python package we will build throughout the workshops
- `test_data/`: Test images and data for the workshops
- `docs/`: Files related to the documentation you are currently reading
