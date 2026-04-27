## Success! Project {{project.name}} is ready to go

Project root: `{{project.root}}`

### Next steps

1. **Define your prompts:** open your prompt definition file,
and add a prompt in the prompt column for each attribute you want
to extract. `{{project.prompt_csv_path}}`

2. **Point your documents to files:** open your link map and add a
filename for the full text you want to link each document to
`{{project.link_map_path}}`

3. **Link documents:** link documents to full texts using the link map. Run

```sh
deet project link
```

4. **Run a data extraction experiment:** Use the wizard to set up a
data extraction experiment using your data. Run

```sh
deet run extract
```

If you prefer to skip the wizard, edit the config file `{{project.config_path}}`
and pass this as a command line argument

```sh
deet run --config-path {{project.config_path}} extract
```
