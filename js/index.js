const carz = {
    main: (() => {

        const input_resize = ((target) => {
            target.style.minHeight = (target.scrollHeight + "px");
        });
        for (let input of document.querySelectorAll(".carz-input-bar > textarea")) {
            input.addEventListener("keyup", ((event) => { input_resize(event.target); }));
            input_resize(input);
        }

    })
};
carz.main();