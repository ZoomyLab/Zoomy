# Quarto project template

This is a template comes with the following features:
- publishing your student offer online with a QR-code for the `student_offer.pdf`
- a central platform for all code/notes/documentation/presentation/organisation
- automatic publishing to public/private webpage (requires tasks from the task list below)
- partially automated installation for Ubuntu/Debian (may require updates for additional dependencies - see wiki)
- basic quarto report/documentation template 
    - html
    - (WIP) pdf 
- basic quarto presentation template 
- basic quarto project management template 
- basic wiki for general intallation tips/guidelines

# Usecases
- student projects (from proposal to project realization)
    - optional public webpage for additional information for your project proposal -> QR code on student_offer
    - single-place for students to code and document and make progress reports for meetings
    - organization tab for writing e.g. minutus or fixing certain goal (project management)
- standalone phd project
    - same as a student project -> just scale it up :)
- multi-project
    - ask me and I can help (@Ingo)
- You only need a component? 
    No problem! Just fork and delete all the stuff you do not like.

## Todos before publishing this project

### Mandetory
- [ ] fork this template to make your own project. Make sure to use select an appropriate group, e.g. student projects -> `mbd_student`. (TODO: a list of available groups and their usage should be posted here.)
- [ ] your make a first commit/push to trigger the pipeline to generate the webpage

### Optional
- [ ] Disable *Unique domain* name to simplify the web adrress: *Deploy > Pages > disable Unique domain*.
- [ ] Create a `student_offer.pdf` (`mbd_sciebo/students/ProjectCalls/templates/student_offer_with_qr.docx`) and copy the pdf into `docs/student_offer.pdf`. Make sure to update the QR code as well. Word has a build in function for that. The link to the new public webpage will be `https://mbd_student.pages.rwth-aachen.de/projects/<your_new_git_repo_name>`, where you need to replace `<your_new_git_repo_name>`.
- [ ] remove the fork relationship. This can be done under *Settings > General > Expand Advanced > Remove fork relationship*. (If you want to create a custom template, **I would recommand not removing the fork relationship**, so you can still pull changes in case this template improves in the future).

### First Steps
- [ ] update the landing page of the project `index.qmd`
- [ ] add python requirements into the `requirementes.txt`
- [ ] update `docs/orga/tasklist.qmd`
- [ ] update `docs/orga/organization.qmd`
- [ ] update `docs/orga/milestones.qmd`
- [ ] add/delete the literature in `docs/references.bib`
- [ ] add any other requirements into the install script `scripts/install.sh`
- [ ] change the contact information in the `_quarto.yml` file

## Todos: improvements template
Please add anything that needs to be changes in this template here, if you do not want to change the template on your own.
- [ ] report: conversion to latex/pdf
    - [ ] report: proper title page (maybe truly latex imported?)
    - [ ] report template: more structure (e.g. for thesis) - like a latex template
- [ ] automatic option to disable *unique domain*
- [ ] add groups for use-cases and describe above in readme


## FAQ:

### Where are the links to the webpage?
- private page (showing the content of `_quarto-private.yml`): `https://<group>.pages.rwth-aachen.de/<project_name>` (if you disable the *Unique domain* option.). Otherwise, check *Deploy > Pages*  
- public page (showing the content of `_quarto-public.yml`): `https://mbd_web.pages.rwth-aachen.de/projects/<project_name>` (or `https://mbd_web.pages.rwth-aachen.de/projects/<project_name>/index.html` if the first one does not work)


