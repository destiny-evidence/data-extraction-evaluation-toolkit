## Welcome to deet

`deet`, the data extraction evaluation toolkit, is a command line app to extract data from documents.

{% if project %}
You are currently in the project directory for: **{{ project.name }}**

Run

```sh
deet project --help
```

for help configuring the project, or

```sh
deet run --help
```

to learn how to extract data from the documents in your project
{% else %}
To start using deet, set up a project by running

```sh
deet project init
```

If you already set up a project before, move into that directory (`cd my-deet-project`), and run `deet` again, to see next steps.
{% endif %}
