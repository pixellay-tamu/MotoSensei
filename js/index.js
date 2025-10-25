const carz = {
    main: (() => {
        const go_button = document.querySelector(".carz-go-button");

        const input_resize = ((target) => {
            target.style.minHeight = (target.scrollHeight + "px");
            if (target.value.length) {
                go_button.removeAttribute("disabled");
            } else {
                go_button.setAttribute("disabled", "");
            }
        });
        const user_input = document.querySelector(".carz-input-bar > textarea");
        user_input.addEventListener("keyup", ((event) => { input_resize(event.target); }));
        user_input.addEventListener("keydown", ((event) => {
            if ((event.key === "Enter") && !event.shiftKey) {
                event.preventDefault();
                go_button.click();
            }
        }));
        input_resize(user_input);

        const upload_input = document.createElement("input");
        upload_input.setAttribute("type", "file");
        upload_input.setAttribute("accept", ".png, .jpeg");
        const on_upload_button_click = (() => {
            upload_input.click();
        });
        document.querySelector(".carz-upload-button").addEventListener("click", on_upload_button_click);

        const chat = document.querySelector(".carz-chat");
        go_button.addEventListener("click", (() => {
            const bubble = document.createElement("div");
            bubble.className = "carz-chat-bubble";
            bubble.setAttribute("data-id", "1");
            bubble.innerText = user_input.value;
            user_input.value = "";

            user_input.style.minHeight = "";
            input_resize(user_input);
            
            chat.appendChild(bubble);

            setTimeout(() => {
                const bubble2 = document.createElement("div");
                bubble2.className = "carz-chat-bubble";
                bubble2.innerText = "[response]";
                chat.appendChild(bubble2);
                bubble2.scrollIntoView();
            }, 500);
        }));

    })
};
carz.main();